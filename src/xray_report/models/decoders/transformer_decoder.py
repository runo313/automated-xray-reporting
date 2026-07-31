#!/usr/bin/env python3
"""
Transformer decoder conditioned on image regions and a findings vector.

The findings vector is projected into embedding space and prepended as a
prefix token, so every generated token can attend to it. Image regions enter
as cross-attention memory.

Changed from the first version:
  - tgt_key_padding_mask is now passed. Self-attention previously attended
    over <pad> positions; the loss masked them but the representations were
    already contaminated, which matters when ~4% of sequences are truncated
    and the rest are padded to max_len.
  - Learned positional embeddings are added to the memory. The 49 spatial
    regions were previously interchangeable as far as the decoder could tell,
    so it could not learn systematic spatial relationships.
  - norm_first=True. Pre-norm transformers are markedly easier to train from
    scratch at this depth than the post-norm default.
  - dim_feedforward and dropout are set explicitly rather than left at the
    2048/0.1 defaults, which were oversized relative to embed_dim=256.
  - padding_idx on the embedding, so <pad> keeps a zero vector.

Place at: src/xray_report/models/decoders/transformer_decoder.py
"""

import torch
import torch.nn as nn

from src.xray_report.models.decoders.positional_encoding import PositionalEncoding


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 num_labels, feature_dim, max_len, dropout=0.1,
                 pad_idx=0, num_regions=49):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout,
                                              max_length=max_len + 8)

        self.findings_proj = nn.Linear(num_labels, embed_dim)
        self.memory_proj = nn.Linear(feature_dim, embed_dim)

        # Without this the 49 regions are permutation-invariant to the decoder.
        self.memory_pos = nn.Parameter(torch.zeros(1, num_regions, embed_dim))
        nn.init.normal_(self.memory_pos, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(embed_dim)
        )
        self.fc = nn.Linear(embed_dim, vocab_size)

    def _memory(self, enc_output, labels):
        mem = self.memory_proj(enc_output)
        mem = mem + self.memory_pos[:, :mem.size(1)]
        # Findings as an extra memory slot, directly attendable from every
        # decoding step rather than only reachable through the prefix token.
        findings_mem = self.findings_proj(labels).unsqueeze(1)
        return torch.cat([mem, findings_mem], dim=1)

    def forward(self, tgt_seq, enc_output, labels):
        """
        Args:
            tgt_seq: (batch, seq_len) input token ids
            enc_output: (batch, num_regions, feature_dim) encoder spatial features
            labels: (batch, num_labels) findings vector

        Returns:
            logits: (batch, seq_len, vocab_size), aligned to predicting
                    tgt_seq shifted one position left
        """
        device = tgt_seq.device

        tgt_embed = self.embedding(tgt_seq)
        tgt_embed = self.pos_encoder(tgt_embed)

        findings_embed = self.findings_proj(labels).unsqueeze(1)
        tgt_embed = torch.cat([findings_embed, tgt_embed], dim=1)

        memory = self._memory(enc_output,labels)

        seq_len = tgt_embed.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )

        # The prepended findings token is never padding, hence the leading
        # column of False. Every row therefore keeps at least one visible
        # position, so no row can produce an all-masked softmax.
        pad_mask = tgt_seq == self.pad_idx
        prefix = torch.zeros(tgt_seq.size(0), 1, dtype=torch.bool, device=device)
        tgt_key_padding_mask = torch.cat([prefix, pad_mask], dim=1)

        decoded = self.transformer_decoder(
            tgt_embed, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.fc(decoded)
        return logits[:, 1:, :]     # drop the findings-prefix position

    def forward_sequence(self, tgt_seq, enc_output, labels):
        """
        Teacher-forced pass over a full target sequence.

        Takes tgt_seq including <bos>/<eos>/padding, handles the input/target
        shift internally, and returns (logits, attn_weights) so both decoders
        share one interface.

        Returns:
            logits: (batch, seq_len - 1, vocab_size), aligned to tgt_seq[:, 1:]
        """
        return self.forward(tgt_seq[:, :-1], enc_output, labels), None

    @torch.no_grad()
    def generate(self, enc_output, labels, max_len, bos_idx, eos_idx,
                 temperature=0.0):
        """
        Greedy (or sampled) decoding, no teacher forcing.

        Recomputes the full prefix at each step rather than caching, which is
        O(n^2) but keeps the code honest. Fine for evaluation on a few
        thousand images; revisit if generation becomes the bottleneck.

        Returns:
            generated: (batch, <= max_len) token ids, including the leading
                       <bos>
        """
        batch_size = enc_output.shape[0]
        device = enc_output.device

        generated = torch.full((batch_size, 1), bos_idx,
                               dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        eos = torch.tensor(eos_idx, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(generated, enc_output, labels)
            step = logits[:, -1, :]

            if temperature and temperature > 0:
                probs = torch.softmax(step / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(1)
            else:
                next_token = step.argmax(dim=-1)

            next_token = torch.where(finished, eos, next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            finished = finished | (next_token == eos_idx)
            if finished.all():
                break

        return generated, None