#!/usr/bin/env python3
"""
Train the report decoder.

The classifier is already trained to 0.790 test macro AUC and validated, so
this loads checkpoints/cls_full/best.pt and FREEZES it by default. The decoder
is then the only thing learning, which makes the ablation arms interpretable
and roughly halves epoch time by skipping the DenseNet backward pass.

Ablation arms, selected with --condition:

    both       image regions + findings vector      the full model
    findings   findings vector only                 learned template
    image      image regions only                   does the encoder carry
                                                    anything the classifier
                                                    missed?
    none       neither                              unconditional language
                                                    model floor

Ablations are implemented by zeroing the unused input rather than by changing
architecture, so all four arms share identical parameter counts and the
comparison is clean.

Scheduled findings sampling: training conditions on ground-truth labels early
and on the classifier's own predictions later, annealing between them. Without
this the decoder learns to trust a signal it never sees at inference, and with
the classifier at 0.79 AUC that gap is not small. Validation and generation
always use predictions.

Validation reports clinical F1 on generated text, not only loss. Loss alone
cannot distinguish a working model from one that has learned the marginal
token distribution, which is exactly what hid the classifier failure earlier.

Usage:
    python3 -m src.xray_report.train_decoder \
        --parquet-path data/chexpert_plus_fixed.parquet \
        --vocab-path data/vocab.pkl \
        --image-root data/images \
        --classifier-checkpoint checkpoints/cls_full/best.pt \
        --checkpoint-dir checkpoints/dec_both \
        --condition both --decoder-type transformer \
        --num-epochs 8 --amp
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from src.xray_report.config import DEFAULT_MAX_LEN, LABEL_COLS, redirect_output
from src.xray_report.dataloader import build_dataloaders
from src.xray_report.eval.baselines import score_reports
from src.xray_report.models.classifier_head import ClassifierHead
from src.xray_report.models.decoders.rnn_attention import AttentionDecoder
from src.xray_report.models.decoders.transformer_decoder import TransformerDecoder
from src.xray_report.models.encoders.cnn_pretrained import PretrainedCNNEncoder
from src.xray_report.models.encoders.vit_pretrained import RadDinoEncoder
from src.xray_report.models.losses import MaskedCrossEntropyLoss
from src.xray_report.utils.vocabulary import load_vocab


def decode_tokens(ids, idx_to_token, eos_idx, bos_idx, pad_idx):
    """Token ids -> string, stopping at <eos> and dropping specials."""
    words = []
    for i in ids:
        i = int(i)
        if i == eos_idx:
            break
        if i in (bos_idx, pad_idx):
            continue
        words.append(idx_to_token.get(i, '<unk>'))
    return ' '.join(words)


class DecoderTrainer:
    def __init__(self, encoder, classifier, decoder, train_loader, val_loader,
                 optimizer, ce_loss, vocab, args, val_df=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        torch.manual_seed(args.random_state)
        np.random.seed(args.random_state)

        self.encoder = encoder.to(self.device)
        self.classifier = classifier.to(self.device)
        self.decoder = decoder.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.ce_loss = ce_loss.to(self.device)
        self.args = args
        self.val_df = val_df

        self.token_to_idx = vocab['token_to_idx']
        self.idx_to_token = vocab['idx_to_token']
        self.bos = self.token_to_idx['<bos>']
        self.eos = self.token_to_idx['<eos>']
        self.pad = self.token_to_idx.get('<pad>', 0)

        self.use_amp = args.amp and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.checkpoint_dir = args.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_metric = -float('inf')
        self.history = []
        self.start_epoch = 0

    # -------------------------------------------------------------- inputs

    def condition_inputs(self, spatial, findings):
        """Zero whichever stream this ablation arm withholds."""
        mode = self.args.condition
        if mode == 'both':
            return spatial, findings
        if mode == 'findings':
            return torch.zeros_like(spatial), findings
        if mode == 'image':
            return spatial, torch.zeros_like(findings)
        if mode == 'none':
            return torch.zeros_like(spatial), torch.zeros_like(findings)
        raise ValueError(f"unknown condition: {mode}")

    def findings_vector(self, labels, cls_logits, epoch, training):
        """
        Ground truth, classifier predictions, or an annealed mix.

        At inference only predictions exist, so validation always uses them.
        """
        pred = torch.sigmoid(cls_logits.float()).detach()

        if not training or self.args.findings_source == 'pred':
            return pred
        if self.args.findings_source == 'gt':
            return labels

        frac = epoch / max(self.args.num_epochs - 1, 1)
        p_gt = (self.args.sched_start +
                (self.args.sched_end - self.args.sched_start) * frac)
        use_gt = (torch.rand(labels.size(0), 1, device=labels.device)
                  < p_gt).float()
        return use_gt * labels + (1.0 - use_gt) * pred

    # -------------------------------------------------------------- loop

    def preflight(self):
        print("\n--- preflight ---")
        images, labels, mask, text_ids = next(iter(self.train_loader))
        images = images.to(self.device)
        labels = labels.to(self.device)
        text_ids = text_ids.to(self.device)

        print(f"images {tuple(images.shape)}  range "
              f"[{images.min():.0f}, {images.max():.0f}]")
        print(f"text_ids {tuple(text_ids.shape)}  "
              f"pad fraction {(text_ids == self.pad).float().mean():.2f}")

        with torch.no_grad():
            pooled, spatial = self.encoder(images)
            cls_logits = self.classifier(pooled)
        print(f"spatial {tuple(spatial.shape)}  pooled {tuple(pooled.shape)}")

        findings = self.findings_vector(labels, cls_logits, 0, True)
        feats, cond = self.condition_inputs(spatial, findings)
        print(f"condition='{self.args.condition}'  "
              f"feats |mean| {feats.abs().mean():.4f}  "
              f"cond |mean| {cond.abs().mean():.4f}")

        logits, _ = self.decoder.forward_sequence(text_ids, feats, cond)
        print(f"decoder logits {tuple(logits.shape)}  "
              f"target {tuple(text_ids[:, 1:].shape)}")
        assert logits.shape[1] == text_ids.shape[1] - 1, \
            "decoder output length does not match the shifted target"

        n_train = sum(p.numel() for p in self.decoder.parameters()
                      if p.requires_grad)
        print(f"trainable decoder params: {n_train/1e6:.2f}M")
        print("preflight passed\n")

    def fit(self):
        self.preflight()
        start = time.time()

        for epoch in range(self.start_epoch, self.args.num_epochs):
            t0 = time.time()
            self.decoder.train()
            self.encoder.train(self.args.unfreeze_encoder)
            self.classifier.train(False)

            running = 0.0
            for images, labels, mask, text_ids in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                text_ids = text_ids.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    if self.args.unfreeze_encoder:
                        pooled, spatial = self.encoder(images)
                    else:
                        with torch.no_grad():
                            pooled, spatial = self.encoder(images)
                    with torch.no_grad():
                        cls_logits = self.classifier(pooled)

                    findings = self.findings_vector(labels, cls_logits,
                                                    epoch, True)
                    feats, cond = self.condition_inputs(spatial, findings)
                    logits, _ = self.decoder.forward_sequence(text_ids,
                                                              feats, cond)
                    loss = self.ce_loss(logits, text_ids[:, 1:])

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g['params']],
                    max_norm=1.0,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                running += loss.item()

            train_loss = running / len(self.train_loader)
            val_loss = self.validate()

            line = (f"Epoch {epoch+1}/{self.args.num_epochs} — "
                    f"gen_loss: {train_loss:.4f} — val_loss: {val_loss:.4f}")

            metric = -val_loss
            clinical = None
            if (epoch + 1) % self.args.gen_eval_every == 0:
                clinical = self.generation_eval()
                if clinical is not None:
                    metric = clinical['macro_f1']
                    line += (f" — val_macro_f1: {clinical['macro_f1']:.4f}"
                             f" — distinct: {clinical['distinct']}")

            line += f" — time: {time.time() - t0:.1f}s"
            print(line)

            self.history.append({
                'epoch': epoch + 1, 'train_loss': train_loss,
                'val_loss': val_loss,
                'macro_f1': clinical['macro_f1'] if clinical else None,
            })

            self.save_checkpoint(epoch + 1, 'last.pt')
            if metric > self.best_metric:
                self.best_metric = metric
                self.save_checkpoint(epoch + 1, 'best.pt')

        print(f"Training complete in {time.time() - start:.1f}s")
        print(f"Best validation metric: {self.best_metric:.4f}")

    @torch.no_grad()
    def validate(self):
        self.decoder.eval()
        self.encoder.eval()
        self.classifier.eval()

        running = 0.0
        for images, labels, mask, text_ids in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            text_ids = text_ids.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                pooled, spatial = self.encoder(images)
                cls_logits = self.classifier(pooled)
                findings = self.findings_vector(labels, cls_logits, 0, False)
                feats, cond = self.condition_inputs(spatial, findings)
                logits, _ = self.decoder.forward_sequence(text_ids, feats, cond)
                loss = self.ce_loss(logits, text_ids[:, 1:])

            running += loss.item()

        return running / len(self.val_loader)

    @torch.no_grad()
    def generation_eval(self):
        """
        Free-running generation on a validation subset, scored with the same
        harness the baselines use. This is the number that actually matters;
        val_loss cannot tell fluent-but-empty from correct.
        """
        if self.val_df is None or self.args.gen_eval_size <= 0:
            return None

        self.decoder.eval()
        self.encoder.eval()
        self.classifier.eval()

        wanted = self.args.gen_eval_size
        texts, rows, seen = [], [], 0

        for images, labels, mask, text_ids in self.val_loader:
            if seen >= wanted:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            pooled, spatial = self.encoder(images)
            cls_logits = self.classifier(pooled)
            findings = self.findings_vector(labels, cls_logits, 0, False)
            feats, cond = self.condition_inputs(spatial, findings)

            gen, _ = self.decoder.generate(feats, cond, self.args.max_len,
                                           self.bos, self.eos)
            for row in gen:
                texts.append(decode_tokens(row.tolist(), self.idx_to_token,
                                           self.eos, self.bos, self.pad))
            rows.extend(range(seen, seen + images.size(0)))
            seen += images.size(0)

        texts = texts[:wanted]
        sub = self.val_df.iloc[:len(texts)]

        print()
        result = score_reports(texts, sub[LABEL_COLS], name="val generation")
        result['distinct'] = len(set(texts))

        print("  sample generations:")
        for t in texts[:3]:
            print(f"    {t[:150]}")
        print()
        return result

    def save_checkpoint(self, epoch, filename):
        ckpt = {
            'epoch': epoch,
            'decoder_state_dict': self.decoder.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_metric': self.best_metric,
            'args': vars(self.args),
            'label_cols': LABEL_COLS,
            'checkpoint_dir': self.checkpoint_dir,
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(ckpt, path)
        print(f"saved checkpoint: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.decoder.load_state_dict(ckpt['decoder_state_dict'])
        self.encoder.load_state_dict(ckpt['encoder_state_dict'])
        self.classifier.load_state_dict(ckpt['classifier_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.history = ckpt.get('history', [])
        self.best_metric = ckpt.get('best_metric', -float('inf'))
        self.start_epoch = ckpt['epoch']
        print(f"resumed from {path} at epoch {self.start_epoch}")


# ------------------------------------------------------------------- main

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Train the report decoder.")
    ap.add_argument('--parquet-path', default='data/chexpert_plus_fixed.parquet')
    ap.add_argument('--vocab-path', default='data/vocab.pkl')
    ap.add_argument('--image-root', default='data/images')
    ap.add_argument('--classifier-checkpoint',
                    default='checkpoints/cls_full/best.pt')
    ap.add_argument('--checkpoint-dir', required=True)
    ap.add_argument('--log-dir', default='logs')
    ap.add_argument('--resume-from', default=None)

    ap.add_argument('--decoder-type', default='transformer',
                    choices=['rnn', 'transformer'])
    ap.add_argument('--condition', default='both',
                    choices=['both', 'findings', 'image', 'none'])
    ap.add_argument('--findings-source', default='scheduled',
                    choices=['gt', 'pred', 'scheduled'])
    ap.add_argument('--sched-start', type=float, default=1.0,
                    help="P(use ground-truth findings) at epoch 0")
    ap.add_argument('--sched-end', type=float, default=0.0,
                    help="P(use ground-truth findings) at the final epoch")

    ap.add_argument('--embed-dim', type=int, default=256)
    ap.add_argument('--num-heads', type=int, default=8)
    ap.add_argument('--num-layers', type=int, default=4)
    ap.add_argument('--hidden-size', type=int, default=512)
    ap.add_argument('--findings-embed-dim', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.1)

    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-epochs', type=int, default=8)
    ap.add_argument('--decoder-lr', type=float, default=3e-4)
    ap.add_argument('--encoder-lr', type=float, default=1e-5)
    ap.add_argument('--weight-decay', type=float, default=1e-2)
    ap.add_argument('--max-len', type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--unfreeze-encoder', action='store_true',
                    help="also fine-tune the encoder; slower and risks the "
                         "validated classifier features")
    ap.add_argument('--train-subsample', type=int, default=None)
    ap.add_argument('--gen-eval-size', type=int, default=500)
    ap.add_argument('--gen-eval-every', type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(
        args.log_dir, f"dec_{args.decoder_type}_{args.condition}_{stamp}.log")
    redirect_output(log_path)
    print(f"logging to {log_path}")
    print(f"args: {json.dumps(vars(args), indent=2)}\n")

    df = pd.read_parquet(args.parquet_path)
    vocab = load_vocab(args.vocab_path)
    token_to_idx = vocab['token_to_idx']
    pad_idx = token_to_idx.get('<pad>', 0)
    print(f"vocab: {len(token_to_idx)} tokens, max_len {args.max_len}")

    loaders = build_dataloaders(
        df, args.image_root, token_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_len=args.max_len,
        train_subsample=args.train_subsample,
    )
    print(f"train={len(loaders['train'].dataset)} "
          f"val={len(loaders['val'].dataset)} "
          f"test={len(loaders['test'].dataset)}\n")

    # val_loader is unshuffled, so the first N rows of the val split line up
    # with the first N generated reports.
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #encoder = PretrainedCNNEncoder()
    encoder = RadDinoEncoder(pool_to=7)
    
    classifier = ClassifierHead(feature_dim=encoder.feature_dim,
                                num_labels=len(LABEL_COLS))

    ckpt = torch.load(args.classifier_checkpoint, map_location=device,
                      weights_only=False)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    classifier.load_state_dict(ckpt['classifier_state_dict'])
    print(f"loaded classifier from {args.classifier_checkpoint} "
          f"(epoch {ckpt.get('epoch')}, val AUC {ckpt.get('best_val_auc')})")

    for p in classifier.parameters():
        p.requires_grad = False
    if not args.unfreeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        print("encoder and classifier frozen; training the decoder only")
    else:
        print("encoder unfrozen; classifier still frozen")

    if args.decoder_type == 'rnn':
        decoder = AttentionDecoder(
            vocab_size=len(token_to_idx),
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            findings_embed_dim=args.findings_embed_dim,
            num_labels=len(LABEL_COLS),
            feature_dim=encoder.feature_dim,
        )
    else:
        decoder = TransformerDecoder(
            vocab_size=len(token_to_idx),
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            num_labels=len(LABEL_COLS),
            feature_dim=encoder.feature_dim,
            max_len=args.max_len,
            dropout=args.dropout,
            pad_idx=pad_idx,
            num_regions=encoder.num_regions,
        )

    ce_loss = MaskedCrossEntropyLoss(pad_idx=pad_idx)

    groups = [{'params': [p for p in decoder.parameters() if p.requires_grad],
               'lr': args.decoder_lr}]
    if args.unfreeze_encoder:
        groups.append({'params': [p for p in encoder.parameters()
                                  if p.requires_grad],
                       'lr': args.encoder_lr})
    optimizer = optim.AdamW(groups, weight_decay=args.weight_decay)

    trainer = DecoderTrainer(
        encoder=encoder, classifier=classifier, decoder=decoder,
        train_loader=loaders['train'], val_loader=loaders['val'],
        optimizer=optimizer, ce_loss=ce_loss, vocab=vocab, args=args,
        val_df=val_df,
    )

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    trainer.fit()