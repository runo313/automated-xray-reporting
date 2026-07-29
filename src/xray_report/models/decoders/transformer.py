#!/usr/bin/env python3
import torch
import torch.nn as nn
from src.xray_report.models.decoders.positional_encoding import PositionalEncoding
from src.xray_report.models.encoders.cnn_pretrained import PretrainedCNNEncoder


class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size, embed_dim, num_heads, num_layers, 
                 num_labels, feature_dim, max_len,dropout=0.1):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_length=max_len)
            self.findings_proj = nn.Linear(num_labels, embed_dim)      # projects findings into the same space as token embeddings
            self.memory_proj = nn.Linear(feature_dim, embed_dim)
            decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self,tgt_seq, enc_output, labels):
        """
            Args:
                tgt_seq: (batch, seq_len) target token ids, full sequence at once
                enc_output: (batch, num_regions, feature_dim) encoder spatial features
                labels: (batch, num_labels) findings vector

            Returns: logits: (batch, seq_len, vocab_size) — predicted distributions
        """
        tgt_embed = self.embedding(tgt_seq)  # (batch, seq_len, embed_dim)
        tgt_embed = self.pos_encoder(tgt_embed) # adds position info
        findings_embed = self.findings_proj(labels).unsqueeze(1)
        tgt_embed = torch.cat([findings_embed, tgt_embed], dim=1)

        memory = self.memory_proj(enc_output)

        seq_len = tgt_embed.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt_embed.device)

        decoded = self.transformer_decoder(tgt_embed, memory, tgt_mask=causal_mask)

        logits = self.fc(decoded)
        logits = logits[:, 1:, :]

        return logits