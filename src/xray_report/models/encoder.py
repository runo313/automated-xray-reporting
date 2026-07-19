"""
Layer 1: CNN image encoder.
Starts from ImageNet-pretrained weights; self-supervised pretraining
variant added later as a separate build_encoder(pretrained="selfsup") path.
"""

import torch.nn as nn


def build_encoder(pretrained: str = "imagenet", backbone: str = "resnet50") -> nn.Module:
    raise NotImplementedError
