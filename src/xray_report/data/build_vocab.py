#!/usr/bin/env python3
"""
Two jobs.

1. MEASURE what fraction of impression text cannot be produced from a single
   frontal radiograph. 95.6% of impressions are unique, and radiology
   impressions routinely reference prior studies, dates, and clinical history.
   None of that is visible in the image. Whatever share this is sets a hard
   ceiling on n-gram metrics and decides whether to filter or reword targets.

2. REBUILD vocab.pkl with normalization that actually works. The current
   tokenizer gives 19.8% OOV; normalizing punctuation and enumerators drops
   that to 0.33% at min_freq=5.

Usage:
    python3 rebuild_text.py --parquet data/chexpert_plus_fixed.parquet \
                            --vocab-out data/vocab.pkl --min-freq 5
"""

import argparse
import collections
import os
import pickle
import re
import shutil

import pandas as pd

TOKEN_RE = re.compile(r"\d+\.\d+|\w+|[^\w\s]")
ENUM_RE = re.compile(r"(?:^|\n)\s*\d+\s*[.):-]\s*")

SPECIALS = ['<pad>', '<bos>', '<eos>', '<unk>']

# Language that can only come from information outside the image.
UNGENERATABLE = {
    'prior comparison': r'\b(compar\w*|previous\w*|prior|interval|since)\b',
    'change over time': r'\b(unchanged|stable|improv\w+|worsen\w+|resolv\w+|'
                        r'increas\w+|decreas\w+|progress\w+)\b',
    'explicit date': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
    'clinical history': r'\b(history|clinical\w*|indication|given the|'
                        r'known\b|reported)\b',
    'recommendation': r'\b(recommend\w*|suggest follow|correlat\w+ clinically|'
                      r'clinical correlation)\b',
}


def normalize(text):
    if not isinstance(text, str):
        return ''
    text = text.replace('\\n', '\n')
    text = ENUM_RE.sub(' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text):
    return TOKEN_RE.findall(normalize(text))


def measure_ungeneratable(series):
    print("=" * 68)
    print("HOW MUCH OF THE TARGET TEXT IS NOT IN THE IMAGE?")
    print("=" * 68)

    norm = series.map(normalize)
    n = len(norm)

    print(f"{'category':<22} {'% of impressions':>18}")
    print("-" * 44)

    any_hit = pd.Series(False, index=norm.index)
    for name, pattern in UNGENERATABLE.items():
        hit = norm.str.contains(pattern, regex=True, na=False)
        any_hit |= hit
        print(f"{name:<22} {hit.mean():>17.1%}")

    print("-" * 44)
    print(f"{'ANY of the above':<22} {any_hit.mean():>17.1%}")

    clean = norm[~any_hit]
    print(f"\nimpressions with none of it: {len(clean)} ({1 - any_hit.mean():.1%})")
    if len(clean):
        lens = clean.map(lambda t: len(TOKEN_RE.findall(t)))
        print(f"  median length {lens.median():.0f} tokens "
              f"(vs {norm.map(lambda t: len(TOKEN_RE.findall(t))).median():.0f} overall)")

    print("""
Read this as your realistic ceiling. Text referencing a prior study, an
interval change, or clinical history cannot be produced from one frontal
image, so a decoder either hallucinates it or is penalised for omitting it.
Either way n-gram scores are capped, which is the argument for judging on
clinical-efficacy metrics (does the report state the right findings?)
rather than on BLEU.

Two options if this share is large:
  - keep everything, report BLEU with the caveat, judge on CheXbert F1
  - add a filtered subset as a second evaluation slice, so you can show
    performance on the image-derivable portion separately
Filtering the TRAINING set is a bigger call: it removes the examples that
teach hedging language, and the model still has to handle them at test time.
""")
    return any_hit


def build_vocab(texts, min_freq):
    counter = collections.Counter(t for txt in texts for t in tokenize(txt))
    kept = [w for w, c in counter.most_common() if c >= min_freq]

    token_to_idx = {tok: i for i, tok in enumerate(SPECIALS)}
    for w in kept:
        if w not in token_to_idx:
            token_to_idx[w] = len(token_to_idx)

    idx_to_token = {i: t for t, i in token_to_idx.items()}
    return token_to_idx, idx_to_token, counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', default='data/chexpert_plus_fixed.parquet')
    ap.add_argument('--text-col', default='section_impression')
    ap.add_argument('--vocab-out', default='data/vocab.pkl')
    ap.add_argument('--min-freq', type=int, default=5)
    ap.add_argument('--max-len', type=int, default=96)
    ap.add_argument('--measure-only', action='store_true')
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    train = df[df['split'] == 'train'] if 'split' in df.columns else df
    print(f"{len(df)} rows, {len(train)} train\n")

    measure_ungeneratable(train[args.text_col].fillna(''))

    if args.measure_only:
        return

    # ------------------------------------------------------------- vocab
    print("=" * 68)
    print("REBUILDING VOCABULARY")
    print("=" * 68)

    t2i, i2t, counter = build_vocab(train[args.text_col].fillna(''),
                                    args.min_freq)
    print(f"distinct tokens in train: {len(counter)}")
    print(f"vocab at min_freq={args.min_freq}: {len(t2i)} "
          f"(including {len(SPECIALS)} specials)")

    held = df[df['split'] == 'val'][args.text_col].fillna('')
    toks = [tokenize(t) for t in held]
    total = sum(len(x) for x in toks)
    oov = sum(t not in t2i for x in toks for t in x)
    print(f"OOV on val: {oov / max(total, 1):.2%}")

    lens = [len(x) for x in toks]
    over = sum(l > args.max_len - 2 for l in lens) / max(len(lens), 1)
    print(f"truncated at max_len={args.max_len}: {over:.1%}")

    if os.path.exists(args.vocab_out):
        backup = args.vocab_out + '.bak'
        shutil.copy2(args.vocab_out, backup)
        print(f"\nbacked up existing vocab to {backup}")

    with open(args.vocab_out, 'wb') as f:
        pickle.dump({'token_to_idx': t2i, 'idx_to_token': i2t}, f)
    print(f"wrote {args.vocab_out}")

    print(f"""
Now make these match, or the new vocab will not be used:

  1. config.py:  DEFAULT_MAX_LEN = {args.max_len}
  2. utils/vocabulary.py: replace tokenize() with the normalize + TOKEN_RE
     version from this file. datasets.py calls it, so nothing else changes.
  3. Delete stale caches: find . -name __pycache__ -exec rm -rf {{}} +
""")


if __name__ == '__main__':
    main()
