#!/usr/bin/env python3
"""
Losses.

Changed from the original:
  - compute_pos_weight now derives counts from config.encode_label_matrix, the
    same function the dataset uses. Previously it counted only explicit 0.0
    entries while the dataset fed blanks to the loss as confident negatives,
    so the true negative-to-positive ratio was far higher than the weight it
    returned. On rare findings that gap was roughly an order of magnitude,
    which pushes the model toward predicting all-negative.
  - Weights are clipped. An unclipped ratio on a finding with a few hundred
    positives produces a weight in the hundreds, which destabilises training
    without buying recall.
  - MaskedBCELoss keeps its original normalisation, which was already correct:
    dividing by mask.sum() rather than numel() makes losses comparable across
    batches with different amounts of masking.
"""

import numpy as np
import torch
import torch.nn as nn

from src.xray_report.config import LABEL_COLS, encode_label_matrix


class MaskedBCELoss(nn.Module):
    """
    Binary cross-entropy over the label matrix, with masked positions excluded.

    mask == 0 entries contribute nothing to the loss or the gradient, which is
    how the U-Ignore uncertainty policy and (optionally) unmentioned findings
    are kept out of training.
    """

    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, logits, labels, mask):
        per_element_loss = self.bce(logits, labels)    # (batch, num_labels)
        masked_loss = per_element_loss * mask
        return masked_loss.sum() / mask.sum().clamp(min=1)


def compute_pos_weight(df, label_cols=None, max_weight=25.0, verbose=True,
                       **policy_kwargs):
    """
    Per-label positive-class weights for BCEWithLogitsLoss.

    Counts are taken over the encoded label matrix, so what gets counted as a
    negative is exactly what the loss will treat as a negative. Passing a
    different blank_policy or uncertainty_policy here than the dataset uses
    reintroduces the original bug, so pass the same kwargs or none at all.

    Args:
        df: Training split only. Computing this over val or test leaks.
        label_cols: Label order. Defaults to config.LABEL_COLS.
        max_weight: Clip. Findings with very few positives otherwise produce
                    weights large enough to destabilise training.
        verbose: Print the per-label counts and weights.

    Returns:
        Tensor of shape (num_labels,), float32.
    """
    label_cols = list(label_cols or LABEL_COLS)
    labels, mask = encode_label_matrix(df, label_cols=label_cols, **policy_kwargs)

    n_valid = mask.sum(axis=0)
    n_pos = (labels * mask).sum(axis=0)
    n_neg = n_valid - n_pos

    weights = n_neg / np.maximum(n_pos, 1.0)
    clipped = np.minimum(weights, max_weight)

    if verbose:
        print(f"\n--- pos_weight (max_weight={max_weight}) ---")
        print(f"{'label':<30} {'n_pos':>8} {'n_neg':>9} {'raw':>8} {'used':>8}")
        for i, col in enumerate(label_cols):
            flag = "  clipped" if clipped[i] < weights[i] else ""
            print(f"{col:<30} {int(n_pos[i]):>8} {int(n_neg[i]):>9} "
                  f"{weights[i]:>8.2f} {clipped[i]:>8.2f}{flag}")

        sparse = [label_cols[i] for i in range(len(label_cols)) if n_pos[i] < 500]
        if sparse:
            print(f"\nfewer than 500 positives: {', '.join(sparse)}")
            print("AUC on these will be noisy. Report them, but do not tune on them.")
        print()

    return torch.tensor(clipped, dtype=torch.float32)


class MaskedCrossEntropyLoss(nn.Module):
    """
    Token-level cross-entropy that ignores padding positions.
    Used for the decoder: padded target positions contribute nothing.
    """

    def __init__(self, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, seq_len, vocab_size) decoder output logits.
            targets: (batch, seq_len) ground-truth token ids.

        Returns:
            Scalar loss averaged over non-pad token positions only.
        """
        batch, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(batch * seq_len, vocab_size)
        targets_flat = targets.reshape(batch * seq_len)

        per_token_loss = self.ce(logits_flat, targets_flat)
        mask = (targets_flat != self.pad_idx).float()
        masked_loss = per_token_loss * mask
        return masked_loss.sum() / mask.sum().clamp(min=1)