#!/usr/bin/env python3
import torch
import torch.nn as nn
"""
Multi-label findings classifier.
Takes an encoder's pooled feature vector and produces one raw logit per finding.
"""

class ClassifierHead(nn.Module):
    def __init__(self, feature_dim, num_labels, hidden_dim=512, dropout=0.3):
        """
        feature_dim: Length of the pooled feature vector from the encoder.
        num_labels: Number of findings to predict (14, in this project).
        hidden_dim: Size of the intermediate hidden layer.
        dropout: Dropout probability applied after the hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )
    def forward(self,pooled):
        """
        Args:
            pooled: Tensor of shape (batch_size, feature_dim)

        Returns:Tensor of shape (batch_size, num_labels), raw
        (pre-sigmoid) scores — pass through BCEWithLogitsLoss
        during training, or torch.sigmoid() at inference time.
        """
        return self.net(pooled)
        