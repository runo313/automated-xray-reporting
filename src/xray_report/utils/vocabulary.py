#!/usr/bin/env python3
"""
Tokenization, vocabulary building, and text encoding/decoding for the report-generation decoder.
Provides functions to tokenize radiology impression text, build a
word-level vocabulary from a training corpus, and encode/decode text
to and from fixed-length integer sequences.
"""
import pandas as pd
import re
import argparse
import pickle

PUNCT_TO_KEEP = ['.', ',', '(', ')', ':', '/', '-', '"']

def read_parquet_train_text(parquet_path):
    """
     Load the merged labels/text parquet file and return the training split's section_impression text as a list of strings.

     Args:
        parquet_path (str or Path): Path to the merged parquet file
            containing at least 'split' and 'section_impression' columns.

    Returns:
        list[str]: Raw impression text for every row where split == 'train'.
    """
    data= pd.read_parquet(parquet_path)
    train_set = data[(data['split']=='train')]
    return train_set['section_impression'].values.tolist()
    


def tokenize(text):
    """
    Tokenize a single radiology impression string into a list of word and punctuation tokens.

    Args:
        text (str): Raw section_impression text for one report.

    Returns:
        list[str]: Ordered list of lowercase word and punctuation tokens.
    """
    text = text.lower().strip()
    text = text.replace('\n', ' ')
    text = text.replace('&lt;deleted&gt;', ' ')

    # pad each punctuation char with spaces so it splits off as its own token
    for p in PUNCT_TO_KEEP:
        text = text.replace(p, f' {p} ')

    # collapse multiple spaces down to one
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split(' ')


def build_vocab(min_freq,tokenized_text):
    """
    Build a word-level vocabulary from tokenized text.
    Assigns indices to every token that appears at least `min_freq` times.
    """
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
    token_to_idx= {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
    idx_to_token = {token:index for index, token in token_to_idx.items()}
    counter = {}
    for tokens in tokenized_text:
        for token in tokens:
            counter[token] = counter.get(token,0)+1

    for token, freq in sorted(counter.items()):
        if freq >= min_freq and token not in token_to_idx:
            idx= len(token_to_idx)
            token_to_idx[token] = idx
            idx_to_token[idx] = token
    print(f"token size: {len(token_to_idx)}")
    return {'token_to_idx': token_to_idx, 'idx_to_token': idx_to_token}

def encode(tokens, token_to_idx, max_len):
    """
     Convert a list of tokens into a fixed-length list of vocabulary indices.
    """
    ids = [token_to_idx.get('<bos>')]
    for tok in tokens:
        ids.append(token_to_idx.get(tok, token_to_idx['<unk>']))
    ids.append(token_to_idx['<eos>'])

    if len(ids) > max_len:
        ids = ids[:max_len - 1] + [token_to_idx['<eos>']]  # keep eos even after truncation
    else:
        ids = ids + [token_to_idx['<pad>']] * (max_len - len(ids))
    return ids

def decode(ids, idx_to_token):
    """Convert a list of vocabulary indices back into a readable string."""
    tokens = []
    for i in ids:
        token = idx_to_token[i]
        if token == '<eos>':
            break
        if token in ('<pad>', '<bos>'):
            continue
        tokens.append(token)
    return ' '.join(tokens)

def save_vocab(vocab, path):
    """Save a vocabulary dict to disk via pickle."""
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)


def load_vocab(path):
    """Load a vocabulary dict previously saved with save_vocab."""
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save a vocabulary from training report text.")
    parser.add_argument('--parquet-path', required=True, help="Path to merged labels parquet file")
    parser.add_argument('--vocab-out', required=True, help="Path to save the built vocabulary (pickle)")
    parser.add_argument('--min-freq', type=int, default=5, help="Minimum token frequency to include in vocab")
    args = parser.parse_args()

    src_train = read_parquet_train_text(args.parquet_path)
    train_tokens = [tokenize(t) for t in src_train]
    print(f"Total sentence: {len(train_tokens)}")
    vocab = build_vocab(args.min_freq, train_tokens)
    save_vocab(vocab, args.vocab_out)
    print(f"vocab saved to {args.vocab_out}")


