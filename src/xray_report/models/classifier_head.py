import torch
import torch.nn as nn
import torchxrayvision as xrv
from src.xray_report.models.encoders.base import BaseEncoder

class PretrainedCNNEncoder(BaseEncoder, nn.Module):
    def __init__(self, freeze_backbone=False):
        super().__init__()
        # Load DenseNet-121 pretrained specifically on CheXpert radiographs
        self.densenet = xrv.models.DenseNet(weights="densenet121-res224-chex")
        self.backbone = self.densenet.features
        self.feature_dim = 1024
        self.num_regions = 49

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images):
        x = self.backbone(images)           # (batch, 1024, 7, 7)
        x = torch.relu(x)                   # DenseNet feature activation
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.feature_dim, self.num_regions)
        spatial_output = x.permute(0, 2, 1)  # (batch, 49, 1024)
        pooled_output = spatial_output.mean(dim=1)   # (batch, 1024)
        return pooled_output, spatial_output