#!/usr/bin/env python3
from abc import ABC, abstractmethod

class BaseEncoder(ABC):
    """
    Shared interface every image encoder variant must implement.
    This is what lets the CNN-from-scratch, pretrained CNN, and pretrained ViT encoders be swapped
    """
    feature_dim: int   # length of each feature vector (pooled and per-region)
    num_regions: int   # number of spatial regions in the grid output

    def forward(self,images):
        """
        Args:
            images: Tensor of shape (batch_size, 3, H, W)

        Returns:
            pooled: Tensor of shape (batch_size, feature_dim)
                A single summary vector per image, for the classification head.
            spatial: Tensor of shape (batch_size, num_regions, feature_dim)
                A per-region feature grid, for the attention decoder.
        """
        raise NotImplementedError