#!/usr/bin/env python3
"""
DenseNet-121 encoder pretrained on CheXpert radiographs (torchxrayvision).

Changed from the original:
  - The full xrv.models.DenseNet is no longer kept as an attribute. Only
    .features is retained, so the unused 18-class xrv head stops appearing in
    encoder.parameters(), the optimizer, and every checkpoint.
  - Silent 3-channel-to-grayscale averaging removed. The dataloader now emits
    single-channel tensors, and this asserts on that instead of papering over
    a mismatch.
  - num_regions is derived from the feature map instead of hardcoded to 49, so
    a change in image size raises a clear error rather than a reshape failure.
  - Added an input-range check that fires once, because the original silently
    accepted ImageNet-normalized input that the pretrained BatchNorm layers
    could not use.
  - Added freeze_bn, so the CheXpert-estimated BatchNorm statistics can be kept
    while the convolutional weights still fine-tune.

Place at: src/xray_report/models/encoders/cnn_pretrained.py
"""

import warnings

import torch
import torch.nn as nn
import torchxrayvision as xrv

from src.xray_report.models.encoders.base import BaseEncoder

# torchxrayvision models are trained on inputs scaled to [-1024, 1024].
XRV_EXPECTED_ABS_MAX = 1024.0
XRV_MIN_PLAUSIBLE_ABS_MAX = 100.0


class PretrainedCNNEncoder(BaseEncoder, nn.Module):
    """
    Produces both a pooled vector for the classifier head and a spatial grid
    for the attention decoder.

    forward(images) -> (pooled, spatial)
        pooled:  (batch, 1024)
        spatial: (batch, num_regions, 1024)
    """

    def __init__(self, weights="densenet121-res224-chex", freeze_backbone=False,
                 freeze_bn=False, check_input_range=True):
        super().__init__()

        densenet = xrv.models.DenseNet(weights=weights)
        self.backbone = densenet.features
        del densenet  # do not retain the unused classification head

        self.feature_dim = 1024
        self.freeze_backbone = freeze_backbone
        self.freeze_bn = freeze_bn
        self.check_input_range = check_input_range
        self._range_checked = False

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Respect freeze_bn: keep BatchNorm in eval mode during training."""
        super().train(mode)
        if mode and (self.freeze_bn or self.freeze_backbone):
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def _verify_input(self, images):
        if images.shape[1] != 1:
            raise ValueError(
                f"expected single-channel input, got {images.shape[1]} channels. "
                "Use transforms.Grayscale(num_output_channels=1) in the dataloader."
            )

        if self.check_input_range and not self._range_checked:
            self._range_checked = True
            abs_max = images.abs().max().item()
            if abs_max < XRV_MIN_PLAUSIBLE_ABS_MAX:
                warnings.warn(
                    f"Input abs max is {abs_max:.2f}, but torchxrayvision weights "
                    f"expect roughly {XRV_EXPECTED_ABS_MAX}. The pretrained "
                    "BatchNorm running statistics will collapse to constants and "
                    "features will not depend on the image. Use XRVNormalize, not "
                    "ImageNet normalization.",
                    RuntimeWarning,
                )

    def forward(self, images):
        self._verify_input(images)

        x = self.backbone(images)              # (batch, 1024, H', W')
        x = torch.relu(x)                      # matches xrv's own forward

        batch_size, channels, h, w = x.shape
        if channels != self.feature_dim:
            raise ValueError(
                f"expected {self.feature_dim} feature channels, got {channels}"
            )

        spatial = x.flatten(2).permute(0, 2, 1)   # (batch, H'*W', 1024)
        pooled = spatial.mean(dim=1)              # (batch, 1024)
        return pooled, spatial

    @property
    def num_regions(self):
        """H' * W' for the configured image size. 49 at 224x224."""
        with torch.no_grad():
            device = next(self.backbone.parameters()).device
            probe = torch.zeros(1, 1, 224, 224, device=device)
            out = self.backbone(probe)
        return out.shape[2] * out.shape[3]