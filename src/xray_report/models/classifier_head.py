"""
Layer 2: multi-label findings classification head on top of encoder features.
"""

import torch.nn as nn


class FindingsClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_findings: int):
        super().__init__()
        raise NotImplementedError
