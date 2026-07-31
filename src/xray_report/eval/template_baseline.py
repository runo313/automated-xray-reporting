#!/usr/bin/env python3
"""
Template baseline: turn classifier predictions into a structured report.

This is Stage 2. It trains nothing. The classifier already predicts 14
findings at 0.790 test macro AUC; this converts those predictions into
sentences and scores the result through the same harness used for retrieval,
so the baselines are directly comparable.

It is also the structured findings summary the project promises, so it is a
deliverable rather than only a control.

FIXED from the first version. tune_thresholds used to select rows with
truth.isin([0.0, 1.0]), keeping only entries where CheXbert wrote an explicit
value. On this data a blank IS the implicit negative, so that subset runs
75-98% positive and the tuner rewarded predicting positive for everything:
nine of fourteen thresholds collapsed to the 0.05 floor and the generated
reports asserted ten findings at once. Thresholds now come from
config.encode_label_matrix, the same convention the training loss uses.

Also fixed: NEGATIVE['Consolidation'] already contained the word "no", which
produced "No no focal consolidation, ...".

Thresholds are tuned on VALIDATION, never on test.

Usage:
    python3 -m src.xray_report.eval.template_baseline \
        --checkpoint checkpoints/cls_full/best.pt \
        --test-size 2000
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torchvision import transforms

from src.xray_report.config import LABEL_COLS, encode_label_matrix
from src.xray_report.eval.baselines import load_encoder, score_reports

SPLIT_PREFIXES = ('train/', 'valid/', 'val/', 'test/')

# Phrasing is chosen to be unambiguously extractable by text_labeler, so the
# baseline measures the classifier rather than the wording. Each string uses a
# surface form that appears in that labeler's PHRASES list.
POSITIVE = {
    'Enlarged Cardiomediastinum': 'The cardiomediastinal silhouette is enlarged.',
    'Cardiomegaly':               'There is cardiomegaly.',
    'Lung Opacity':               'There is a pulmonary opacity.',
    'Lung Lesion':                'There is a pulmonary nodule.',
    'Edema':                      'There is pulmonary edema.',
    'Consolidation':              'There is consolidation.',
    'Pneumonia':                  'There is pneumonia.',
    'Atelectasis':                'There is atelectasis.',
    'Pneumothorax':               'There is a pneumothorax.',
    'Pleural Effusion':           'There is a pleural effusion.',
    'Pleural Other':              'There is pleural thickening.',
    'Fracture':                   'There is a fracture.',
    'Support Devices':            'Support devices are in place.',
}

# Findings a radiologist routinely rules out explicitly. Emitting these makes
# the output read like a real impression; they do not affect positive-class F1.
# Values are bare noun phrases: the leading "No" is added when rendering.
ROUTINE_NEGATIVES = ['Consolidation', 'Pleural Effusion', 'Pneumothorax']
NEGATIVE = {
    'Consolidation':    'focal consolidation',
    'Pleural Effusion': 'pleural effusion',
    'Pneumothorax':     'pneumothorax',
}

NORMAL_REPORT = 'No acute cardiopulmonary abnormality.'


def strip_prefix(p):
    for prefix in SPLIT_PREFIXES:
        if p.startswith(prefix):
            return p[len(prefix):]
    return p


class XRVNormalize:
    def __call__(self, x):
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        return (2.0 * x - 1.0) * 1024.0


def eval_transform(size=224):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        XRVNormalize(),
    ])


@torch.no_grad()
def predict(paths, image_root, encoder, classifier, device, batch_size=64):
    """Sigmoid probabilities per label, plus the paths that loaded."""
    tf = eval_transform()
    probs, kept = [], []

    for i in range(0, len(paths), batch_size):
        imgs, ok = [], []
        for rel in paths[i:i + batch_size]:
            full = os.path.join(image_root, strip_prefix(rel))
            try:
                with Image.open(full) as im:
                    imgs.append(tf(im))
                ok.append(rel)
            except Exception:
                pass
        if not imgs:
            continue
        pooled, _ = encoder(torch.stack(imgs).to(device))
        logits = classifier(pooled)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        kept.extend(ok)

    return np.concatenate(probs), kept


def tune_thresholds(probs, labels, label_cols):
    """
    Per-label threshold maximising F1 on the validation split.

    Uses encode_label_matrix so blanks count as negatives, matching both the
    training loss and the scoring harness. Filtering to explicit values here
    is what produced the 0.05 thresholds in the first version.
    """
    truth_arr, mask_arr = encode_label_matrix(
        labels.reset_index(drop=True), label_cols
    )

    print("=" * 66)
    print("THRESHOLDS (tuned on validation)")
    print("=" * 66)
    print(f"{'label':<30} {'thresh':>8} {'val F1':>8} {'prev':>8}")
    print("-" * 58)

    thresholds = {}
    for i, col in enumerate(label_cols):
        keep = mask_arr[:, i] == 1.0
        y = truth_arr[keep, i].astype(int)
        p = probs[keep, i]

        best_t, best_f1 = 0.5, 0.0
        if len(y) >= 20 and y.sum() > 0:
            for t in np.arange(0.05, 0.96, 0.025):
                f1 = f1_score(y, (p > t).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_t, best_f1 = float(t), f1

        thresholds[col] = best_t
        prev = y.mean() if len(y) else 0.0
        print(f"{col:<30} {best_t:>8.3f} {best_f1:>8.3f} {prev:>8.3f}")

    print("\nA threshold pinned at the 0.05 floor with prevalence near 1.0 "
          "means\nthe evaluation subset is degenerate; check the label "
          "convention.\n")
    return thresholds


def render(prob_row, label_cols, thresholds, include_negatives=True):
    """Turn one row of probabilities into a report string."""
    positive = [c for i, c in enumerate(label_cols)
                if c in POSITIVE and prob_row[i] > thresholds[c]]

    # Support Devices alone is not a pathology, so it does not suppress the
    # normal-study statement.
    pathologies = [c for c in positive if c != 'Support Devices']

    if not pathologies:
        parts = [NORMAL_REPORT]
        if 'Support Devices' in positive:
            parts.append(POSITIVE['Support Devices'])
        return ' '.join(parts)

    parts = [POSITIVE[c] for c in label_cols if c in positive]

    if include_negatives:
        absent = [NEGATIVE[c] for c in ROUTINE_NEGATIVES
                  if c not in positive and c in NEGATIVE]
        if absent:
            if len(absent) == 1:
                parts.append(f"No {absent[0]}.")
            else:
                parts.append("No " + ", ".join(absent[:-1]) +
                             f", or {absent[-1]}.")

    return ' '.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/cls_full/best.pt')
    ap.add_argument('--parquet', default='data/chexpert_plus_fixed.parquet')
    ap.add_argument('--image-root', default='data/images')
    ap.add_argument('--text-col', default='section_impression')
    ap.add_argument('--val-size', type=int, default=3000)
    ap.add_argument('--test-size', type=int, default=2000)
    ap.add_argument('--no-negatives', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_parquet(args.parquet)

    encoder, classifier = load_encoder(args.checkpoint, device)

    # ---------------------------------------------------------- thresholds
    val = df[df['split'] == 'val'].head(args.val_size).reset_index(drop=True)
    print(f"\ntuning thresholds on {len(val)} val images...")
    val_probs, val_kept = predict(val['path_to_image'].tolist(),
                                  args.image_root, encoder, classifier, device)
    val = val.set_index('path_to_image').loc[val_kept].reset_index()
    thresholds = tune_thresholds(val_probs, val[LABEL_COLS], LABEL_COLS)

    # ---------------------------------------------------------- test
    test = df[df['split'] == 'test'].head(args.test_size).reset_index(drop=True)
    print(f"predicting on {len(test)} test images...")
    probs, kept = predict(test['path_to_image'].tolist(),
                          args.image_root, encoder, classifier, device)
    test = test.set_index('path_to_image').loc[kept].reset_index()

    reports = [render(probs[i], LABEL_COLS, thresholds,
                      include_negatives=not args.no_negatives)
               for i in range(len(test))]

    print("\nexample generated reports:")
    for r in reports[:5]:
        print(f"  {r}")

    lengths = [len(r.split()) for r in reports]
    n_findings = [sum(probs[i, j] > thresholds[c]
                      for j, c in enumerate(LABEL_COLS) if c in POSITIVE)
                  for i in range(len(test))]
    print(f"\nmean length {np.mean(lengths):.1f} words, "
          f"{len(set(reports))} distinct of {len(reports)}")
    print(f"mean findings asserted per report: {np.mean(n_findings):.2f}\n")

    score_reports(reports, test[LABEL_COLS],
                  references=test[args.text_col].fillna('').tolist(),
                  name="template baseline")

    if 'has_prior_ref' in test.columns:
        clean = ~test['has_prior_ref'].astype(bool)
        if clean.sum() >= 300:
            print(f"image-derivable slice ({int(clean.sum())} references)\n")
            score_reports([r for r, k in zip(reports, clean) if k],
                          test.loc[clean, LABEL_COLS],
                          references=test.loc[clean, args.text_col]
                                         .fillna('').tolist(),
                          name="template, image-derivable slice")

    print("""
Sanity check before believing any of this: mean findings asserted per report
should be roughly 2 to 4. Ten means the thresholds collapsed and the metric
is rewarding indiscriminate positives.

Compare macro clinical F1 against retrieval. If the template wins, the
classifier carries more clinical signal than nearest neighbour lookup does,
and a learned decoder has to beat BOTH to justify itself. If it loses, the
classifier is the weak link rather than the generation step.

A template can produce at most 2^13 distinct outputs against 95.6% unique
references, so low diversity is expected and is exactly the limitation a
learned decoder should overcome.
""")


if __name__ == '__main__':
    main()
