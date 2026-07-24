#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision
from src.xray_report.models.encoders.base import BaseEncoder

class PretrainedCNNEncoder(BaseEncoder, nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_dim = 2048
        self.num_regions = 49

    def forward(self,images):
        x = self.backbone(images) # (batch, 2048, 7, 7)
        batch_size = x.shape[0]
        x=x.reshape(batch_size,self.feature_dim,self.num_regions) # (batch, 2048, 49)
        spatial_output = x.permute(0, 2, 1) # (batch, 49, 2048)
        pooled_output= spatial_output.mean(dim=1)  # (batch, 2048)
        return pooled_output, spatial_output

