#!/usr/bin/env python3
import torch
import torch.nn as nn
"""
Multi-label findings classifier.
Takes an encoder's pooled feature vector and produces one raw logit per finding.
"""

class ClassifierHead(nn.Module):
    def __init__(self, feature_dim, num_labels, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )
    def forward(self,pooled):
        return self.net(pooled)
        