"""
Layer 3: findings-conditioned attention decoder.
GRU-based to start; transformer decoder variant is a later, separate class.
"""

import torch.nn as nn


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, findings_dim: int):
        super().__init__()
        raise NotImplementedError
