#!/usr/bin/env python3
import torch
import torch.nn as nn


class MaskedBCELoss(nn.Module):
    """
    Masked positions (mask == 0) are excluded from the loss entirely,
    correctly implementing the U-Ignore policy for uncertain (-1.0) labels.
    """

    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, logits, labels, mask):
        per_element_loss = self.bce(logits, labels)         # (batch, num_labels), unreduced
        masked_loss = per_element_loss * mask                # zero out ignored entries
        return masked_loss.sum() / mask.sum().clamp(min=1)

def compute_pos_weight(df, label_cols):
    """
    Compute per-label positive-class weights for BCEWithLogitsLoss, based on
    the ratio of negatives to positives for each label in the given dataframe.
    """
    weights = []
    for col in label_cols:
        col_vals = df[col]
        num_pos = (col_vals == 1.0).sum()
        num_neg = (col_vals == 0.0).sum()
        weight = num_neg / max(num_pos, 1)   # avoid divide-by-zero for any label with 0 positives
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)