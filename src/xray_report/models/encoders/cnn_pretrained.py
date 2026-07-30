#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from src.xray_report.models.encoders.base import BaseEncoder

class PretrainedCNNEncoder(BaseEncoder, nn.Module):
    def __init__(self,freeze_backbone=True):
        super().__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.backbone = densenet.features
        self.feature_dim = 1024
        self.num_regions = 49

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self,images):
        x = self.backbone(images) # (batch, 1024, 7, 7)
        batch_size = x.shape[0]
        x=x.reshape(batch_size,self.feature_dim,self.num_regions) # (batch, 1024, 49)
        spatial_output = x.permute(0, 2, 1) # (batch, 49, 1024)
        pooled_output= spatial_output.mean(dim=1)  # (batch, 1024)
        return pooled_output, spatial_output

