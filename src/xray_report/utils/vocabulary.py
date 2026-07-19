"""
Vocabulary class for the report decoder.

Follows the same fixed-special-token-index pattern used in the HW3
seq2seq assignment: <pad>=0, <bos>=1, <eos>=2, <unk>=3.
Fill in build_vocab() once IU X-Ray report text is available.
"""

from collections import Counter


class Vocabulary:
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.token_to_idx = {self.PAD: 0, self.BOS: 1, self.EOS: 2, self.UNK: 3}
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}

    def build_vocab(self, tokenized_reports: list[list[str]]):
        """
        tokenized_reports: list of token lists, one per report.
        TODO: implement once IU X-Ray report text is loaded and tokenized.
        """
        counter = Counter(tok for report in tokenized_reports for tok in report)
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token_to_idx:
                idx = len(self.token_to_idx)
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
        return self

    def __len__(self):
        return len(self.token_to_idx)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_idx.get(t, self.token_to_idx[self.UNK]) for t in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.idx_to_token.get(i, self.UNK) for i in indices]
