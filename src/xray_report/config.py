#!/usr/bin/env python3
"""
Shared config and the single source of truth for label encoding.

The label policy lives here because datasets.py and losses.py both need it and
must not drift. The original bug: datasets.py treated blank as a confident
negative while compute_pos_weight counted only explicit zeros, so positives on
rare findings were under-weighted by roughly an order of magnitude.

Place at: src/xray_report/config.py
"""

import sys

import numpy as np
import pandas as pd

LABEL_COLS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices', 'No Finding',
]

DEFAULT_MAX_LEN = 50
DEFAULT_MIN_FREQ = 5
DEFAULT_IMAGE_SIZE = 224

# --------------------------------------------------------------- label policy

# What to do with a blank / NaN entry, meaning the labeler found no mention.
#   'negative' - treat as a confident 0. CheXpert convention. Keeps the matrix
#                dense, but every blank becomes a training signal.
#   'ignore'   - mask out of the loss. Far more conservative; on CheXpert this
#                masks the majority of entries for most findings.
BLANK_POLICY = 'negative'

# What to do with an explicit uncertain (-1.0) entry, per label.
#   'ones'   - U-Ones. Uncertainty here usually means probably present.
#   'zeros'  - U-Zeros. Uncertainty here is usually hedging.
#   'ignore' - mask out of the loss.
# Standard CheXpert baseline assignments; anything unlisted defaults to ignore.
UNCERTAINTY_POLICY = {
    'Atelectasis': 'ones',
    'Edema': 'ones',
    'Pleural Effusion': 'ones',
    'Cardiomegaly': 'zeros',
    'Consolidation': 'zeros',
}


def encode_label_matrix(df, label_cols=None, blank_policy=BLANK_POLICY,
                        uncertainty_policy=None):
    """
    Turn raw CheXpert label columns into (labels, mask) float32 arrays.

    mask == 1.0 means the entry contributes to the loss.
    mask == 0.0 means it is excluded entirely.

    Both the dataset and compute_pos_weight call this, so the loss and its
    class weighting can never disagree about what a blank means.

    Returns:
        labels: (n_rows, n_labels) float32, values in {0.0, 1.0}
        mask:   (n_rows, n_labels) float32, values in {0.0, 1.0}
    """
    label_cols = list(label_cols or LABEL_COLS)
    uncertainty_policy = uncertainty_policy or UNCERTAINTY_POLICY

    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"label columns not in dataframe: {missing}")

    raw = (
        df[label_cols]
        .apply(pd.to_numeric, errors='coerce')
        .to_numpy(dtype=np.float32)
    )

    labels = np.zeros_like(raw)
    mask = np.zeros_like(raw)

    is_nan = np.isnan(raw)
    is_pos = raw == 1.0
    is_neg = raw == 0.0
    is_unc = raw == -1.0

    labels[is_pos] = 1.0
    mask[is_pos] = 1.0
    mask[is_neg] = 1.0

    if blank_policy == 'negative':
        mask[is_nan] = 1.0          # labels already 0.0 there
    elif blank_policy != 'ignore':
        raise ValueError(f"unknown blank_policy: {blank_policy}")

    for i, col in enumerate(label_cols):
        policy = uncertainty_policy.get(col, 'ignore')
        col_unc = is_unc[:, i]
        if policy == 'ones':
            labels[col_unc, i] = 1.0
            mask[col_unc, i] = 1.0
        elif policy == 'zeros':
            mask[col_unc, i] = 1.0
        elif policy != 'ignore':
            raise ValueError(f"unknown uncertainty policy for {col}: {policy}")

    return labels, mask


def label_stats(df, label_cols=None, **kwargs):
    """
    Per-label summary of what the loss will actually see.

    Print this before training. If a label has a handful of positives, no
    amount of pos_weight will make its AUC meaningful, and you should not
    tune on it.
    """
    label_cols = list(label_cols or LABEL_COLS)
    labels, mask = encode_label_matrix(df, label_cols, **kwargs)

    rows = []
    for i, col in enumerate(label_cols):
        valid = mask[:, i].sum()
        pos = (labels[:, i] * mask[:, i]).sum()
        rows.append({
            'label': col,
            'n_valid': int(valid),
            'n_pos': int(pos),
            'prevalence': float(pos / valid) if valid else 0.0,
            'n_masked': int(len(labels) - valid),
        })

    return pd.DataFrame(rows)


def redirect_output(log_path):
    """Redirect stdout and stderr to the same log file."""
    log_file = open(log_path, 'w', buffering=1)   # line-buffered for tail -f
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description="Inspect the label matrix.")
    ap.add_argument('--parquet-path', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--blank-policy', default=BLANK_POLICY,
                    choices=['negative', 'ignore'])
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet_path)
    if 'split' in df.columns and args.split != 'all':
        df = df[df['split'] == args.split]

    print(f"split={args.split}  rows={len(df)}  blank_policy={args.blank_policy}\n")
    print(label_stats(df, blank_policy=args.blank_policy).to_string(index=False))