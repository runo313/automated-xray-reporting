#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import pandas as pd

from src.xray_report.config import LABEL_COLS, DEFAULT_MAX_LEN
from src.xray_report.dataloader import build_dataloaders
from src.xray_report.utils.vocabulary import load_vocab

from src.xray_report.models.encoders.cnn_pretrained import PretrainedCNNEncoder
from src.xray_report.models.classifier_head import ClassifierHead
from src.xray_report.models.decoders.rnn_attention import AttentionDecoder
from src.xray_report.models.losses import MaskedBCELoss, MaskedCrossEntropyLoss, compute_pos_weight

class XRayReportModel(nn.Module):
    """
    Combines the image encoder, findings classifier, and attention decoder into a single trainable model.
    """
    def __init__(self, encoder, classifier, decoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.decoder = decoder

    def forward(self, images, tgt_seq, labels):
        """
        Full forward pass: image to findings logits and to generated-text logits

        Args:
            images: (batch, 3, H, W)
            tgt_seq: (batch, seq_len) — ground-truth token ids
            labels: (batch, num_labels) — ground-truth findings, used to condition the decoder during training

        Returns:
            classifier_logits: (batch, num_labels)
            decoder_logits: (batch, seq_len - 1, vocab_size)
            attn_weights: (batch, seq_len - 1, num_regions)
        """
        pooled, spatial = self.encoder(images)
        classifier_logits = self.classifier(pooled)

        decoder_logits, attn_weights = self.decoder.forward_sequence(tgt_seq, spatial, labels)

        return classifier_logits, decoder_logits, attn_weights

