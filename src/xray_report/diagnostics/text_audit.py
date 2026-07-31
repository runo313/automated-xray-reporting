#!/usr/bin/env python3
"""
The report text pipeline is unusable as it stands: 19.8% OOV and 31.5% of
impressions truncated at max_len=50. Both come from tokenizing with
text.lower().split(), which makes 'consolidation' and 'consolidation,'
different tokens, turns '1.' into vocabulary, and fuses literal \\n onto
words as '\\nno'.

This measures what normalization buys before you commit to rebuilding
anything, and prints the numbers needed to choose max_len and min_freq.

Usage:
    python3 text_audit.py --parquet data/chexpert_plus_fixed.parquet \
                          --vocab data/vocab.pkl
"""

import argparse
import collections
import pickle
import re

import numpy as np
import pandas as pd

# Split punctuation off words, keep decimals in measurements intact.
TOKEN_RE = re.compile(r"\d+\.\d+|\w+|[^\w\s]")

# Leading enumerators: '1.', '2)', '3 -' at the start of a line or after \n.
ENUM_RE = re.compile(r"(?:^|\n)\s*\d+\s*[.):-]\s*")


def normalize(text):
    """Undo the specific damage visible in the raw impressions."""
    if not isinstance(text, str):
        return ''

    # The data carries literal backslash-n as two characters, not newlines.
    text = text.replace('\\n', '\n')
    text = ENUM_RE.sub(' ', text)          # drop '1.' '2)' enumerators
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)       # collapse all whitespace
    return text.strip()


def tokenize(text):
    return TOKEN_RE.findall(text)


def naive_tokenize(text):
    """What the current pipeline does."""
    return str(text).lower().split()


def vocab_at(counter, min_freq):
    return {w for w, c in counter.items() if c >= min_freq}


def oov_rate(token_lists, vocab):
    total = hits = 0
    for toks in token_lists:
        total += len(toks)
        hits += sum(t not in vocab for t in toks)
    return hits / max(total, 1)


def length_table(lengths, cutoffs=(50, 64, 80, 96, 128)):
    lengths = np.asarray(lengths)
    print(f"  median {np.median(lengths):.0f}   "
          f"p90 {np.percentile(lengths, 90):.0f}   "
          f"p95 {np.percentile(lengths, 95):.0f}   "
          f"p99 {np.percentile(lengths, 99):.0f}   "
          f"max {lengths.max()}")
    print("  truncation by max_len (allowing 2 slots for bos/eos):")
    for c in cutoffs:
        frac = (lengths > c - 2).mean()
        print(f"    {c:>4}: {frac:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', default='data/chexpert_plus_fixed.parquet')
    ap.add_argument('--vocab', default='data/vocab.pkl')
    ap.add_argument('--text-col', default='section_impression')
    ap.add_argument('--sample', type=int, default=50000)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    train = df[df['split'] == 'train'] if 'split' in df.columns else df
    held = df[df['split'] == 'val'] if 'split' in df.columns else df

    raw_train = train[args.text_col].fillna('')
    raw_held = held[args.text_col].fillna('').head(args.sample)

    # ------------------------------------------------------------ current
    print("=" * 66)
    print("CURRENT PIPELINE (lower().split(), existing vocab.pkl)")
    print("=" * 66)

    try:
        with open(args.vocab, 'rb') as f:
            existing = set(pickle.load(f)['token_to_idx'])
        print(f"vocab size: {len(existing)}")
        held_naive = [naive_tokenize(t) for t in raw_held]
        print(f"OOV rate on held-out: {oov_rate(held_naive, existing):.1%}")
        length_table([len(t) for t in held_naive])
    except Exception as e:
        print(f"could not load {args.vocab}: {e}")

    # ------------------------------------------------------------ normalized
    print()
    print("=" * 66)
    print("AFTER NORMALIZATION (punctuation split, enumerators dropped)")
    print("=" * 66)

    norm_train = raw_train.map(normalize)
    norm_held = raw_held.map(normalize)

    train_toks = [tokenize(t) for t in norm_train]
    held_toks = [tokenize(t) for t in norm_held]

    counter = collections.Counter(t for toks in train_toks for t in toks)
    print(f"distinct tokens in train: {len(counter)}")

    print("\n  min_freq   vocab    OOV(held-out)")
    for mf in (1, 2, 3, 5, 10, 20):
        v = vocab_at(counter, mf)
        print(f"  {mf:>8}   {len(v):>6}   {oov_rate(held_toks, v):>10.2%}")

    print("\nlength distribution:")
    length_table([len(t) for t in held_toks])

    # ------------------------------------------------------------ duplicates
    print()
    print("=" * 66)
    print("REPETITION (how strong is the constant-output baseline?)")
    print("=" * 66)

    raw_share = raw_train.str.strip().str.lower().value_counts(normalize=True)
    norm_share = norm_train.value_counts(normalize=True)

    print(f"top string share, raw:        {raw_share.iloc[0]:.2%}")
    print(f"top string share, normalized: {norm_share.iloc[0]:.2%}")
    print(f"top-10 share, normalized:     {norm_share.head(10).sum():.2%}")
    print(f"top-100 share, normalized:    {norm_share.head(100).sum():.2%}")
    print(f"distinct impressions:         {len(norm_share)} "
          f"of {len(norm_train)} rows")

    print("\nmost common normalized impressions:")
    for text, share in norm_share.head(5).items():
        print(f"  {share:6.2%}  {text[:110]}")

    print("""
Read the top-10 and top-100 shares as your mode-collapse risk. If the top
100 normalized impressions cover a large fraction of training data, a model
that memorises a handful of templates will score well on n-gram metrics
while being clinically useless, and you need clinical-efficacy metrics
rather than BLEU to see it.
""")

    # ------------------------------------------------------------ subword
    print("=" * 66)
    print("SUBWORD ALTERNATIVE")
    print("=" * 66)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('bert-base-uncased')
        sub = [tok.tokenize(t) for t in norm_held.head(5000)]
        print(f"vocab size: {tok.vocab_size}   OOV: 0.00% by construction")
        length_table([len(s) for s in sub])
        print("\nSubword tokenization eliminates OOV entirely and is what a")
        print("pretrained decoder would need later. Cost: sequences run")
        print("longer, and vocab.pkl plus both decoders need rewiring.")
    except ImportError:
        print("transformers not installed; skipping.")
        print("pip install transformers to compare.")


if __name__ == '__main__':
    main()
