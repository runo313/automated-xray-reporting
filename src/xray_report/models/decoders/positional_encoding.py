#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout,max_length=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).reshape(-1,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
        self.register_buffer('encoding', encoding)
    def forward(self, x):
        return self.dropout(x + self.encoding[:, :x.size(1)].detach())