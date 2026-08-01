#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on the held-out test set.

Changed from the original:
  - Decoder evaluation is gated. The original ran it unconditionally, including
    on classifier-only checkpoints where it had just printed that the decoder
    weights were uninitialized, producing meaningless BLEU.
  - Checkpoint provenance is printed and verified. The previous run trained into
    checkpoints/classifier_xrv_densenet/ and evaluated
    checkpoints/classifier_densenet_finetune/best.pt from an earlier run at a
    different epoch. Nothing in the code objected.
  - Label ordering stored in the checkpoint is checked against LABEL_COLS. A
    silent permutation here gives exactly chance AUC on every label.
  - Prevalence and support are reported alongside AUC, plus baseline F1 for a
    constant predictor. If your F1 matches the constant baseline, the model has
    learned the label distribution and nothing else.
  - torch.load is called with weights_only=False explicitly, so the choice is
    deliberate rather than a FutureWarning.

Place at: src/xray_report/models/evaluate.py
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.metrics import f1_score, roc_auc_score

from src.xray_report.config import DEFAULT_MAX_LEN, LABEL_COLS, redirect_output
from src.xray_report.dataloader import build_dataloaders
from src.xray_report.models.classifier_head import ClassifierHead
from src.xray_report.models.decoders.rnn_attention import AttentionDecoder
from src.xray_report.models.decoders.transformer_decoder import TransformerDecoder
from src.xray_report.models.encoders.cnn_pretrained import PretrainedCNNEncoder
from src.xray_report.models.encoders.vit_pretrained import RadDinoEncoder
from src.xray_report.models.model import XRayReportModel
from src.xray_report.utils.vocabulary import decode, load_vocab


def load_checkpoint(path, device):
    """Load a checkpoint and report exactly what it is."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint at {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    print(f"\n--- checkpoint ---")
    print(f"path:        {path}")
    print(f"epoch:       {ckpt.get('epoch', 'UNKNOWN')}")
    print(f"saved at:    {ckpt.get('saved_at', 'UNKNOWN')}")
    print(f"written by:  {ckpt.get('checkpoint_dir', 'UNKNOWN')}")
    print(f"best val AUC:{ckpt.get('best_val_auc', 'UNKNOWN')}")

    written_dir = ckpt.get('checkpoint_dir')
    if written_dir and os.path.normpath(written_dir) != os.path.normpath(os.path.dirname(path)):
        print(f"\n  WARNING: this checkpoint was written by a run targeting")
        print(f"  {written_dir}, but you are loading it from")
        print(f"  {os.path.dirname(path)}. Confirm this is the run you meant.")

    ckpt_labels = ckpt.get('label_cols')
    if ckpt_labels and list(ckpt_labels) != list(LABEL_COLS):
        raise ValueError(
            "label ordering mismatch between checkpoint and config.\n"
            f"  checkpoint: {list(ckpt_labels)}\n"
            f"  config:     {list(LABEL_COLS)}\n"
            "A permutation here produces chance AUC on every label."
        )

    print("------------------\n")
    return ckpt


def evaluate_classifier(model, test_loader, device, label_cols):
    """Per-label AUC and F1, with prevalence and a constant-predictor baseline."""
    model.eval()
    all_logits, all_labels, all_masks = [], [], []

    with torch.no_grad():
        for images, labels, mask, _ in test_loader:
            images = images.to(device, non_blocking=True)
            pooled, _ = model.encoder(images)
            logits = model.classifier(pooled)

            all_logits.append(logits.float().cpu())
            all_labels.append(labels)
            all_masks.append(mask)

    probs = torch.sigmoid(torch.cat(all_logits)).numpy()
    labels = torch.cat(all_labels).numpy()
    masks = torch.cat(all_masks).numpy()

    results = {}
    for i, col in enumerate(label_cols):
        valid = masks[:, i] == 1.0
        y_true = labels[valid, i]
        y_prob = probs[valid, i]

        if len(y_true) < 10 or len(np.unique(y_true)) < 2:
            results[col] = {'auc': None, 'f1': None, 'support': int(valid.sum())}
            continue

        auc = roc_auc_score(y_true, y_prob)

        best_f1, best_thresh = 0.0, 0.5
        for thresh in np.arange(0.05, 0.95, 0.05):
            score = f1_score(y_true, (y_prob > thresh).astype(int), zero_division=0)
            if score > best_f1:
                best_f1, best_thresh = score, thresh

        prevalence = float(y_true.mean())
        # F1 of a predictor that always says positive: 2p / (1 + p).
        baseline_f1 = 2 * prevalence / (1 + prevalence) if prevalence > 0 else 0.0

        results[col] = {
            'auc': auc,
            'f1': best_f1,
            'best_threshold': float(best_thresh),
            'prevalence': prevalence,
            'baseline_f1': baseline_f1,
            'support': int(valid.sum()),
        }

    return results


def print_classifier_results(results):
    print("\n=== Classifier results (per label) ===")
    header = (f"{'label':<30} {'AUC':>7} {'F1':>7} {'base F1':>8} "
              f"{'prev':>6} {'n':>7}")
    print(header)
    print("-" * len(header))

    aucs, suspicious = [], []
    for col, m in results.items():
        if m['auc'] is None:
            print(f"{col:<30} {'--':>7} {'--':>7} {'--':>8} {'--':>6} {m['support']:>7}")
            continue
        aucs.append(m['auc'])
        print(f"{col:<30} {m['auc']:>7.3f} {m['f1']:>7.3f} "
              f"{m['baseline_f1']:>8.3f} {m['prevalence']:>6.3f} {m['support']:>7}")
        if abs(m['f1'] - m['baseline_f1']) < 0.02 and m['auc'] < 0.55:
            suspicious.append(col)

    if aucs:
        print(f"\nmacro AUC: {np.mean(aucs):.4f}")

    if len(suspicious) >= len(results) // 2:
        print("\n  WARNING: F1 matches the always-positive baseline on most labels")
        print("  and AUC is near chance. The model has learned the label")
        print("  distribution, not the images. Check input normalization and")
        print("  that images actually load before interpreting anything here.")


def evaluate_decoder(model, test_loader, device, idx_to_token, label_cols,
                     max_len, bos_idx, eos_idx):
    """BLEU-4 plus a crude finding-mention proxy for clinical efficacy."""
    model.eval()
    smoothing = SmoothingFunction().method1

    bleu_scores = []
    finding_hits = {c: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for c in label_cols}

    with torch.no_grad():
        for images, labels, mask, text_ids in test_loader:
            images = images.to(device, non_blocking=True)
            labels_dev = labels.to(device, non_blocking=True)

            _, spatial = model.encoder(images)
            generated, _ = model.decoder.generate(
                spatial, labels_dev, max_len, bos_idx, eos_idx
            )

            for i in range(images.shape[0]):
                gen_text = decode(generated[i].tolist(), idx_to_token)
                ref_text = decode(text_ids[i].tolist(), idx_to_token)

                bleu_scores.append(sentence_bleu(
                    [ref_text.split()], gen_text.split(),
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smoothing,
                ))

                for j, col in enumerate(label_cols):
                    if mask[i, j] == 0.0:
                        continue
                    present = labels[i, j].item() == 1.0
                    mentioned = col.lower() in gen_text.lower()

                    if present and mentioned:
                        finding_hits[col]['tp'] += 1
                    elif present:
                        finding_hits[col]['fn'] += 1
                    elif mentioned:
                        finding_hits[col]['fp'] += 1
                    else:
                        finding_hits[col]['tn'] += 1

    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    efficacy = {}
    for col, c in finding_hits.items():
        tp, fp, fn = c['tp'], c['fp'], c['fn']
        efficacy[col] = {
            'precision': tp / (tp + fp) if (tp + fp) else None,
            'recall': tp / (tp + fn) if (tp + fn) else None,
        }

    return avg_bleu, efficacy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--parquet-path', required=True)
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--log-dir', default='logs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--decoder-type', default='rnn', choices=['rnn', 'transformer'])
    parser.add_argument('--eval-decoder', action='store_true',
                        help="Only meaningful for checkpoints with trained decoder weights.")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    redirect_output(os.path.join(args.log_dir, f"eval_{timestamp}.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    checkpoint = load_checkpoint(args.checkpoint_path, device)

    df = pd.read_parquet(args.parquet_path)
    vocab = load_vocab(args.vocab_path)
    loaders = build_dataloaders(df, args.image_root, vocab['token_to_idx'],
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)
    print(f"test set: {len(loaders['test'].dataset)} examples")

    #encoder = PretrainedCNNEncoder()
    encoder = RadDinoEncoder(freeze_backbone=args.freeze_backbone, pool_to=7)
    classifier = ClassifierHead(feature_dim=encoder.feature_dim,
                                num_labels=len(LABEL_COLS))

    if args.decoder_type == 'rnn':
        decoder = AttentionDecoder(
            vocab_size=len(vocab['token_to_idx']), embed_dim=256, hidden_size=512,
            findings_embed_dim=64, num_labels=len(LABEL_COLS),
            feature_dim=encoder.feature_dim,
        )
    else:
        decoder = TransformerDecoder(
            vocab_size=len(vocab['token_to_idx']), embed_dim=256, num_heads=8,
            num_layers=4, num_labels=len(LABEL_COLS),
            feature_dim=encoder.feature_dim, max_len=DEFAULT_MAX_LEN,
        )

    model = XRayReportModel(encoder, classifier, decoder).to(device)

    has_decoder = 'model_state_dict' in checkpoint
    if has_decoder:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        print("classifier-only checkpoint: decoder weights are uninitialized")

    cls_results = evaluate_classifier(model, loaders['test'], device, LABEL_COLS)
    print_classifier_results(cls_results)

    if args.eval_decoder and not has_decoder:
        print("\nrefusing to evaluate an untrained decoder. "
              "Drop --eval-decoder or use a joint checkpoint.")
    elif args.eval_decoder:
        avg_bleu, efficacy = evaluate_decoder(
            model, loaders['test'], device, vocab['idx_to_token'], LABEL_COLS,
            max_len=DEFAULT_MAX_LEN,
            bos_idx=vocab['token_to_idx']['<bos>'],
            eos_idx=vocab['token_to_idx']['<eos>'],
        )
        print(f"\n=== Decoder results ===")
        print(f"BLEU-4: {avg_bleu:.4f}")
        print("\nClinical efficacy (per finding):")
        for label, m in efficacy.items():
            print(f"{label}: precision={m['precision']}, recall={m['recall']}")