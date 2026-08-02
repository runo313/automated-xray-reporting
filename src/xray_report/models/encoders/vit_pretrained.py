#!/usr/bin/env python3
"""
RAD-DINO encoder: a DINOv2 ViT-B/14 self-supervised on 882k chest radiographs.

- RAD-DINO expects ImageNet-normalized 3-channel input, not the
     [-1024, 1024] single-channel range the rest of this pipeline uses. Rather
     than adding a second transform path through dataloader.py the encoder
     converts internally and defaults to assuming xrv-range input. Pass
     input_range='imagenet' if you ever feed it already-correct tensors.

- Patch grid depends on input size: 224px gives 16x16 = 256 patches, 518px
     (the pretraining resolution) gives 37x37 = 1369. Both are far more than
     the DenseNet's 49 regions. Default is to pool down to 7x7 so cross-
     attention cost and scale stay comparable to the existing decoder runs;
     set pool_to=None to keep the full grid.
"""

import warnings
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.xray_report.models.encoders.base import BaseEncoder

# ImageNet statistics, what DINOv2-derived models expect.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

XRV_SCALE = 1024.0


class RadDinoEncoder(BaseEncoder, nn.Module):
    """
    Args:
        model_name: HuggingFace id. 'microsoft/rad-dino' is the CXR one

        pool_to: side length of the pooled patch grid, e.g. 7 gives 49

        freeze_backbone: no gradients through the ViT.
        
        input_range: 'xrv' if images arrive in [-1024, 1024] from the existing
            dataloader (the default), 'imagenet' if they are already
            ImageNet-normalized 3-channel tensors, 'unit' for [0, 1].

        check_input_range: warn once if the input does not look like the
            declared range. 
    """

    def __init__(self, model_name='microsoft/rad-dino', pool_to=7,freeze_backbone=False, 
                 input_range='xrv', check_input_range=True):
        super().__init__()

        if input_range not in ('xrv', 'imagenet', 'unit'):
            raise ValueError(f"unknown input_range: {input_range}")

        self.model_name = model_name
        self.pool_to = pool_to
        self.input_range = input_range
        self.check_input_range = check_input_range
        self.freeze_backbone = freeze_backbone
        self._range_checked = False

        self.backbone = AutoModel.from_pretrained(model_name)
        self.feature_dim = self.backbone.config.hidden_size          # 768
        self.patch_size = self.backbone.config.patch_size            # 14

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Registered as buffers so .to(device) moves them with the module.
        self.register_buffer('_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _verify_input(self, images):
        if not self.check_input_range or self._range_checked:
            return
        self._range_checked = True

        abs_max = images.abs().max().item()
        if self.input_range == 'xrv' and abs_max < 100:
            warnings.warn(
                f"input_range='xrv' but the input's max magnitude is "
                f"{abs_max:.2f}. These tensors do not look like the "
                "[-1024, 1024] range. Passing the wrong range here silently "
                "destroys the pretrained features.",
                RuntimeWarning,
            )
        elif self.input_range in ('imagenet', 'unit') and abs_max > 100:
            warnings.warn(
                f"input_range='{self.input_range}' but the input's max "
                f"magnitude is {abs_max:.2f}, which looks like xrv-range "
                "data. Pass input_range='xrv' instead.",
                RuntimeWarning,
            )

    def _to_imagenet(self, images):
        """Convert whatever came in into ImageNet-normalized 3-channel."""
        if self.input_range == 'imagenet':
            x = images
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            return x

        if self.input_range == 'xrv':
            # (2 * p - 1) * 1024  ->  p in [0, 1]
            x = (images / XRV_SCALE + 1.0) / 2.0
        else:                                  # 'unit'
            x = images

        x = x.clamp(0.0, 1.0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"expected 1 or 3 channels, got {x.shape[1]}")

        return (x - self._mean) / self._std


    def forward(self, images):
        """
        Args:
            images: (batch, 1 or 3, H, W)

        Returns:
            pooled:  (batch, feature_dim)
            spatial: (batch, num_regions, feature_dim)
        """
        self._verify_input(images)
        x = self._to_imagenet(images)

        h, w = x.shape[-2:]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(
                f"input {h}x{w} is not divisible by patch size "
                f"{self.patch_size}; use 224 (16x16 grid) or 518 (37x37)"
            )

        out = self.backbone(pixel_values=x)
        tokens = out.last_hidden_state              # (B, 1 + N, D), CLS first
        patches = tokens[:, 1:, :]                  # drop CLS

        gh, gw = h // self.patch_size, w // self.patch_size

        if self.pool_to is not None and self.pool_to < gh:
            # (B, N, D) -> (B, D, gh, gw) -> pool -> (B, R, D)
            grid = patches.transpose(1, 2).reshape(
                patches.size(0), self.feature_dim, gh, gw)
            grid = F.adaptive_avg_pool2d(grid, (self.pool_to, self.pool_to))
            spatial = grid.flatten(2).transpose(1, 2)
        else:
            spatial = patches

        pooled = spatial.mean(dim=1)

        return pooled, spatial

    @property
    def num_regions(self):
        if self.pool_to is not None:
            return self.pool_to * self.pool_to
        raise ValueError(
            "num_regions is only fixed when pool_to is set; with the native "
            "grid it depends on input resolution"
        )
