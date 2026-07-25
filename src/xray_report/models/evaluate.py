#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the held-out test set."""

import argparse
import sys
import os
import time
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from src.xray_report.config import LABEL_COLS, DEFAULT_MAX_LEN, redirect_output
from src.xray_report.dataloader import build_dataloaders
from src.xray_report.utils.vocabulary import load_vocab, decode
from src.xray_report.models.encoders.cnn_pretrained import PretrainedCNNEncoder
from src.xray_report.models.classifier_head import ClassifierHead
from src.xray_report.models.decoders.rnn_attention import AttentionDecoder
from src.xray_report.models.model import XRayReportModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate_classifier(model, test_loader, device, label_cols):
    """Compute per-label AUC and F1 on the test set."""
    model.eval()
    all_logits, all_labels, all_masks = [], [], []

    with torch.no_grad():
        for images, labels, mask, text_ids in test_loader:
            images = images.to(device)
            pooled, _ = model.encoder(images)
            logits = model.classifier(pooled)

            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_masks.append(mask)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    masks = torch.cat(all_masks)
    probs = torch.sigmoid(logits)

    results = {}
    for i, col in enumerate(label_cols):
        valid = masks[:, i] == 1.0
        y_true = labels[valid, i].numpy()
        y_prob = probs[valid, i].numpy()

        if len(set(y_true)) < 2:   # AUC undefined with only one class present
            results[col] = {'auc': None, 'f1': None}
            continue

        auc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, (y_prob > 0.5).astype(int))
        results[col] = {'auc': auc, 'f1': f1}

    return results
def evaluate_decoder(model, test_loader, device, idx_to_token, label_cols, max_len, bos_idx, eos_idx):
    """
    Generate text for the test set and compute BLEU-4 and clinical efficacy.
    
    Clinical efficacy: for each finding, checks whether its name appears in
    the generated text as a rough proxy for whether the model wrote about
    conditions that were actually present. 
    """
    model.eval()
    smoothing = SmoothingFunction().method1

    bleu_scores = []
    finding_hits = {col: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for col in label_cols}

    with torch.no_grad():
        for images, labels, mask, text_ids in test_loader:
            images = images.to(device)
            labels_dev = labels.to(device)

            pooled, spatial = model.encoder(images)
            generated, _ = model.decoder.generate(spatial, labels_dev, max_len, bos_idx, eos_idx)

            for i in range(images.shape[0]):
                gen_text = decode(generated[i].tolist(), idx_to_token)
                ref_text = decode(text_ids[i].tolist(), idx_to_token)

                bleu = sentence_bleu(
                    [ref_text.split()], gen_text.split(),
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smoothing,
                )
                bleu_scores.append(bleu)

                for j, col in enumerate(label_cols):
                    if mask[i, j] == 0.0:
                        continue   # skip uncertain labels, same masking principle as training

                    true_present = labels[i, j].item() == 1.0
                    mentioned = col.lower() in gen_text.lower()

                    if true_present and mentioned:
                        finding_hits[col]['tp'] += 1
                    elif true_present and not mentioned:
                        finding_hits[col]['fn'] += 1
                    elif not true_present and mentioned:
                        finding_hits[col]['fp'] += 1
                    else:
                        finding_hits[col]['tn'] += 1

    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    efficacy_results = {}
    for col, counts in finding_hits.items():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        efficacy_results[col] = {'precision': precision, 'recall': recall}

    return avg_bleu, efficacy_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--parquet-path', required=True)
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--log-dir', default='logs')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.log_dir, f"eval_{timestamp}.log")
    redirect_output(log_path)
    print(f"evaluating checkpoint: {args.checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(args.parquet_path)
    vocab = load_vocab(args.vocab_path)
    loaders = build_dataloaders(df, args.image_root, vocab['token_to_idx'], batch_size=args.batch_size)

    encoder = PretrainedCNNEncoder()
    classifier = ClassifierHead(feature_dim=encoder.feature_dim, num_labels=len(LABEL_COLS))
    decoder = AttentionDecoder(
        vocab_size=len(vocab['token_to_idx']), embed_dim=256, hidden_size=512,
        findings_embed_dim=64, num_labels=len(LABEL_COLS), feature_dim=encoder.feature_dim,
    )
    model = XRayReportModel(encoder, classifier, decoder).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"loaded checkpoint from epoch {checkpoint['epoch']}")

    cls_results = evaluate_classifier(model, loaders['test'], device, LABEL_COLS)
    print("\n=== Classifier results (per label) ===")
    for label, metrics in cls_results.items():
        print(f"{label}: AUC={metrics['auc']}, F1={metrics['f1']}")

    avg_bleu, efficacy_results = evaluate_decoder(
    model, loaders['test'], device, vocab['idx_to_token'], LABEL_COLS,
    max_len=DEFAULT_MAX_LEN, bos_idx=vocab['token_to_idx']['<bos>'], eos_idx=vocab['token_to_idx']['<eos>'],
    )

    print(f"\n=== Decoder results ===")
    print(f"BLEU-4: {avg_bleu:.4f}")
    print("\nClinical efficacy (per finding):")
    for label, metrics in efficacy_results.items():
        print(f"{label}: precision={metrics['precision']}, recall={metrics['recall']}")
