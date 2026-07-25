#!/usr/bin/env python3
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size,feature_dim, attn_dim=None):
        super().__init__()
        attn_dim = attn_dim or hidden_size
        self.W_q = nn.Linear(hidden_size, attn_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, attn_dim, bias=False)
        self.w_v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, keys):
        """
            query: decoder's current hidden state
            keys:  encoder's spatial features

            Returns:
            context: (batch, feature_dim) — attention-weighted region features
            weights: (batch, num_regions) — attention distribution, for visualization
        """
        query  = query.unsqueeze(1)
        scores = self.w_v(torch.tanh(self.W_q(query) + self.W_k(keys))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights
    
class AttentionDecoder(nn.Module):
    def __init__(self,vocab_size, embed_dim, hidden_size,findings_embed_dim,num_labels,feature_dim ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.findings = nn.Linear(num_labels, findings_embed_dim)
        self.attention = Attention(hidden_size,feature_dim)
        self.rnn = nn.GRUCell(embed_dim + feature_dim + findings_embed_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self,tgt_token, hidden, enc_output,findings_embed):
        """
            One decoding step.

            Args:
                tgt_token: (batch,)  token id of the previous word (or <bos>)
                hidden: (batch, hidden_size) previous GRU hidden state
                enc_output: (batch, num_regions, feature_dim) encoder spatial features
                findings_embed: (batch, findings_embed_dim)  pre-projected findings vector

            Returns:
                logits: (batch, vocab_size) distribution over next token
                hidden: (batch, hidden_size) new hidden state
                weights: (batch, num_regions) attention weights, for visualization
        """
        embedded = self.embedding(tgt_token)
        context, weights = self.attention(hidden, enc_output)
        rnn_input = torch.cat([embedded, context, findings_embed], dim=-1)
        hidden = self.rnn(rnn_input, hidden)
        logits = self.fc(hidden)
        return logits, hidden, weights

    def forward_sequence(self, tgt_seq, enc_output, labels):
        """
            Run teacher-forced decoding over a full target sequence.

            Args:
                tgt_seq: (batch, seq_len) token ids, including <bos>/<eos>/padding
                enc_output: (batch, num_regions, feature_dim) encoder spatial features
                labels: (batch, num_labels) findings vector (classifier predictions
                    or ground-truth labels, depending on training strategy)

            Returns:
                logits: (batch, seq_len - 1, vocab_size) predicted distributions
                    at each step
                attn_weights: (batch, seq_len - 1, num_regions) attention weights per step, for visualization
        """
        batch_size, seq_len = tgt_seq.shape
        device = tgt_seq.device
        findings_embed = self.findings(labels)
        hidden = torch.zeros(batch_size, self.rnn.hidden_size, device=device)

        all_logits = []
        all_weights = []

        for t in range(seq_len - 1):
            input_token = tgt_seq[:, t]                             
            logits, hidden, weights = self.forward(input_token, hidden, enc_output, findings_embed)
            all_logits.append(logits)
            all_weights.append(weights)

        logits = torch.stack(all_logits, dim=1)                     
        attn_weights = torch.stack(all_weights, dim=1)
        
        return logits, attn_weights