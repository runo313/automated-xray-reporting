#!/usr/bin/env python3
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size,feature_dim, attn_dim=None):
        super().__init__()
        attn_dim = attn_dim or hidden_size
        self.W_q = nn.Linear(hidden_size, attn_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, attn_dim, bias=False)
        self.w_v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, keys):
        """
            query: decoder's current hidden state
            keys:  encoder's spatial features

            Returns:
            context: (batch, feature_dim) — attention-weighted region features
            weights: (batch, num_regions) — attention distribution, for visualization
        """
        query  = query.unsqueeze(1)
        scores = self.w_v(torch.tanh(self.W_q(query) + self.W_k(keys))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights