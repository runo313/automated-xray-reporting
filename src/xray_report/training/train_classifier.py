#!/usr/bin/env python3
"""
Classifier-only training: encoder + classifier head, no decoder.
  - Validation now reports macro AUC, not just loss. Loss alone cannot
    distinguish a working model from one that has learned only the base rates,
    which is exactly what hid the previous failure for five epochs.
  - Best checkpoint is selected on val AUC. On imbalanced multi-label data, BCE
    loss and ranking quality diverge.
  - Checkpoints record the run directory, args, label order, and metrics, so
    evaluation can verify it loaded the model it thinks it did.
  - A first-batch diagnostic prints input range and feature variance before
    training starts. Both failure modes from the previous run are visible here
    within seconds instead of after 90 minutes.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from src.xray_report.config import LABEL_COLS, redirect_output
from src.xray_report.dataloader import build_dataloaders
from src.xray_report.models.classifier_head import ClassifierHead
from src.xray_report.models.encoders.cnn_pretrained import PretrainedCNNEncoder
from src.xray_report.models.encoders.vit_pretrained import RadDinoEncoder
from src.xray_report.models.losses import MaskedBCELoss, compute_pos_weight
from src.xray_report.utils.vocabulary import load_vocab


def per_label_auc(probs, labels, masks, label_cols):
    """AUC per label, skipping labels with no valid examples of both classes."""
    results = {}
    for i, col in enumerate(label_cols):
        valid = masks[:, i] == 1.0
        y_true = labels[valid, i]
        y_prob = probs[valid, i]
        if len(y_true) < 10 or len(np.unique(y_true)) < 2:
            results[col] = None
            continue
        results[col] = roc_auc_score(y_true, y_prob)
    return results


def macro_auc(auc_dict):
    scores = [v for v in auc_dict.values() if v is not None]
    return float(np.mean(scores)) if scores else float('nan')


class ClassifierFitModel:
    def __init__(self, n_iter, encoder, classifier, train_loader, val_loader,
                 optimizer, bce_loss, label_cols, random_state=42,
                 checkpoint_dir=None, use_amp=False, scheduler=None,
                 run_args=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.n_iter = n_iter
        self.encoder = encoder.to(self.device)
        self.classifier = classifier.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.bce_loss = bce_loss.to(self.device)
        self.label_cols = label_cols
        self.run_args = run_args or {}

        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.checkpoint_dir = checkpoint_dir
        self.best_val_auc = -float('inf')
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.history = []

    # diagnostics

    def preflight(self):
        """
        Verify the data actually reaches the model in usable form.

        Three checks:
        input range, per-image variance, and whether encoder features differ
        across images at all.
        """
        print("\n--- preflight ---")
        images, labels, mask, _ = next(iter(self.train_loader))
        images = images.to(self.device)

        print(f"batch shape:  {tuple(images.shape)}   (want (B, 1, 224, 224))")
        print(f"input range:  min={images.min():.1f} max={images.max():.1f} "
              f"mean={images.mean():.1f}")
        print(f"              expected min near -1024, max near 1024")

        per_image_std = images.reshape(images.shape[0], -1).std(dim=1)
        print(f"per-image std: min={per_image_std.min():.1f} "
              f"median={per_image_std.median():.1f}   (near 0 means blank images)")

        self.encoder.eval()
        with torch.no_grad():
            pooled, _ = self.encoder(images)
        across = pooled.std(dim=0).mean().item()
        print(f"feature std across images: {across:.4f}")

        label_frac = (labels == 1.0).float().mean(dim=0)
        print(f"positive rate in batch: min={label_frac.min():.3f} "
              f"max={label_frac.max():.3f}")

        problems = []
        if images.shape[1] != 1:
            problems.append("input is not single-channel")
        if images.abs().max() < 100:
            problems.append("input scale far below the [-1024, 1024] xrv range")
        if per_image_std.min() < 1.0:
            problems.append("at least one image is blank")
        if across < 1e-3:
            problems.append("encoder features are constant across images")

        if problems:
            raise RuntimeError(
                "preflight failed:\n  - " + "\n  - ".join(problems) +
                "\nFix these before training; they produce AUC 0.50 on every label."
            )

        print("preflight passed\n")


    def fit(self):
        self.preflight()
        start_time = time.time()

        for epoch in range(self.n_iter):
            epoch_start = time.time()

            self.encoder.train()
            self.classifier.train()
            running_loss = 0.0

            for images, labels, mask, _ in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    pooled, _ = self.encoder(images)
                    logits = self.classifier(pooled)
                    loss = self.bce_loss(logits, labels, mask)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g['params']], max_norm=1.0,)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            val_loss, val_aucs = self.validate()
            val_macro = macro_auc(val_aucs)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{self.n_iter} — cls_loss: {avg_loss:.4f} — "
                  f"val_loss: {val_loss:.4f} — val_macro_auc: {val_macro:.4f} — "
                  f"time: {epoch_time:.1f}s")

            named = sorted(
                ((c, a) for c, a in val_aucs.items() if a is not None),
                key=lambda t: -t[1],
            )
            print("    " + "  ".join(f"{c}={a:.3f}" for c, a in named[:5]))

            self.history.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'val_macro_auc': val_macro,
                'val_aucs': val_aucs,
            })

            if epoch == 0 and val_macro < 0.55:
                print("\n    WARNING: macro AUC near chance after epoch 1. The model")
                print("    is predicting base rates. Stop and diagnose rather than")
                print("    letting this run to completion.\n")

            if self.checkpoint_dir:
                self.save_checkpoint(epoch + 1, "last.pt")
                if val_macro > self.best_val_auc:
                    self.best_val_auc = val_macro
                    self.save_checkpoint(epoch + 1, "best.pt")

        print(f"Training complete in {time.time() - start_time:.1f}s")
        print(f"Best val macro AUC: {self.best_val_auc:.4f}")

    def validate(self):
        self.encoder.eval()
        self.classifier.eval()

        running_loss = 0.0
        all_probs, all_labels, all_masks = [], [], []

        with torch.no_grad():
            for images, labels, mask, _ in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels_d = labels.to(self.device, non_blocking=True)
                mask_d = mask.to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    pooled, _ = self.encoder(images)
                    logits = self.classifier(pooled)
                    loss = self.bce_loss(logits, labels_d, mask_d)

                running_loss += loss.item()
                all_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
                all_labels.append(labels.numpy())
                all_masks.append(mask.numpy())

        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        masks = np.concatenate(all_masks)

        return running_loss / len(self.val_loader), per_label_auc(
            probs, labels, masks, self.label_cols
        )

    def save_checkpoint(self, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'label_cols': self.label_cols,
            'history': self.history,
            'best_val_auc': self.best_val_auc,
            'checkpoint_dir': self.checkpoint_dir,
            'args': self.run_args,
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"saved checkpoint: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifier-only pretraining (encoder + classifier head).")
    parser.add_argument('--parquet-path', required=True)
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--log-dir', default='logs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--encoder-lr', type=float, default=1e-4)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--freeze-bn', action='store_true',help="Keep CheXpert BatchNorm stats while fine-tuning weights.")
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--amp', action='store_true',help="Mixed precision. Roughly halves epoch time on a T4.")
    parser.add_argument('--train-subsample', type=int, default=None, help="Cap training rows for fast iteration.")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.log_dir, f"classifier_only_{timestamp}.log")
    redirect_output(log_path)
    print(f"logging to {log_path}")
    print(f"args: {json.dumps(vars(args), indent=2)}")

    df = pd.read_parquet(args.parquet_path)
    vocab = load_vocab(args.vocab_path)

    loaders = build_dataloaders(
        df, args.image_root, vocab['token_to_idx'],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
        train_subsample=args.train_subsample,
    )
    print(f"train={len(loaders['train'].dataset)} "
          f"val={len(loaders['val'].dataset)} "
          f"test={len(loaders['test'].dataset)}")

    # encoder = PretrainedCNNEncoder(
    #     freeze_backbone=args.freeze_backbone,
    #     freeze_bn=args.freeze_bn,
    # )
    encoder = RadDinoEncoder(freeze_backbone=args.freeze_backbone, pool_to=7)
    classifier = ClassifierHead(
        feature_dim=encoder.feature_dim,
        num_labels=len(LABEL_COLS),
    )

    train_df = df[df['split'] == 'train']
    pos_weight = compute_pos_weight(train_df, LABEL_COLS)
    bce_loss = MaskedBCELoss(pos_weight=pos_weight)

    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    head_params = [p for p in classifier.parameters() if p.requires_grad]
    print(f"trainable tensors: encoder={len(encoder_params)} head={len(head_params)}")

    groups = [{'params': head_params, 'lr': args.head_lr}]
    if encoder_params:
        groups.insert(0, {'params': encoder_params, 'lr': args.encoder_lr})

    optimizer = optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    trainer = ClassifierFitModel(
        n_iter=args.num_epochs,
        encoder=encoder,
        classifier=classifier,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        optimizer=optimizer,
        bce_loss=bce_loss,
        label_cols=LABEL_COLS,
        random_state=args.random_state,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=args.amp,
        scheduler=scheduler,
        run_args=vars(args),
    )
    trainer.fit()