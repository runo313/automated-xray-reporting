#!/usr/bin/env python3
"""
CheXpertDataset.

Changed from the original:
  - Label encoding moved to config.encode_label_matrix and precomputed once at
    init. Previously each __getitem__ ran pandas .astype/.replace/.fillna on a
    per-row Series, which is slow enough to matter across 200k images, and its
    NaN handling disagreed with compute_pos_weight.
  - The FileNotFoundError fallback that recursed into the next row is gone.
    Missing files now raise by default, with a count reported at init, so you
    know how much of your dataset is actually on disk.
  - Images open as 'L' rather than 'RGB'. The source radiographs are grayscale
    and the transform converts anyway.
  - Path stripping is unchanged, since it already matched download.py.

Place at: src/xray_report/datasets.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from src.xray_report.config import (
    BLANK_POLICY,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_LEN,
    LABEL_COLS,
    UNCERTAINTY_POLICY,
    encode_label_matrix,
)
from src.xray_report.utils.vocabulary import encode, load_vocab, tokenize

SPLIT_PREFIXES = ('train/', 'valid/', 'val/', 'test/')


def strip_split_prefix(raw_path):
    """
    Match how download.py wrote files to disk: one leading split directory
    removed, because images from every split share a single image_root.
    """
    for prefix in SPLIT_PREFIXES:
        if raw_path.startswith(prefix):
            return raw_path[len(prefix):]
    return raw_path


class CheXpertDataset(Dataset):
    def __init__(self, df, image_root, token_to_idx, transform=None,
                 max_len=DEFAULT_MAX_LEN, label_cols=None,
                 text_col='section_impression', path_col='path_to_image',
                 blank_policy=BLANK_POLICY, uncertainty_policy=None,
                 strict=True, verify_paths=True):
        """
        Args:
            df: Dataframe with path_to_image, label columns, and section_impression.
            image_root: Root folder containing the image files.
            token_to_idx: Vocabulary mapping from vocabulary.build_vocab / load_vocab.
            transform: torchvision transform pipeline applied to each image.
            max_len: Fixed length to pad/truncate encoded text to.
            label_cols: Label order. Must match the classifier head's output order.
            blank_policy: 'negative' or 'ignore'. See config.
            uncertainty_policy: Per-label dict for -1.0 entries. See config.
            strict: Raise on a missing or unreadable image instead of substituting.
            verify_paths: Check every path exists at init and report the count.
        """
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.token_to_idx = token_to_idx
        self.transform = transform
        self.max_len = max_len
        self.label_cols = list(label_cols or LABEL_COLS)
        self.text_col = text_col
        self.strict = strict

        # Resolve every path once, up front.
        self.image_paths = [
            os.path.join(image_root, strip_split_prefix(p))
            for p in self.df[path_col]
        ]

        if verify_paths:
            missing = [p for p in self.image_paths if not os.path.exists(p)]
            if missing:
                msg = (f"{len(missing)} of {len(self.image_paths)} images are "
                       f"missing from {image_root}\n"
                       f"  first few: {missing[:3]}")
                if strict:
                    raise FileNotFoundError(msg)
                print(f"WARNING: {msg}")

        # Encode all labels once, via the same function compute_pos_weight uses.
        labels, mask = encode_label_matrix(
            self.df,
            label_cols=self.label_cols,
            blank_policy=blank_policy,
            uncertainty_policy=uncertainty_policy or UNCERTAINTY_POLICY,
        )
        self.labels = torch.from_numpy(labels)
        self.mask = torch.from_numpy(mask)

        # Encode all text once. At max_len=50 this is a few MB even at 200k rows.
        texts = self.df[text_col].fillna('').tolist()
        self.text_ids = torch.tensor(
            np.stack([encode(tokenize(t), token_to_idx, max_len) for t in texts]),
            dtype=torch.long,
        )

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Return one training example as (image, labels, mask, text_ids)."""
        path = self.image_paths[idx]

        try:
            with Image.open(path) as img:
                image = img.convert('L')
                if self.transform:
                    image = self.transform(image)
                else:
                    image = torch.from_numpy(np.array(image, dtype=np.float32))[None]
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            if self.strict:
                raise RuntimeError(
                    f"could not read image for row {idx}: {path}\n"
                    f"  original: {self.df.iloc[idx]['path_to_image']}\n"
                    f"  error: {e}\n"
                    "Not substituting another row. A silently swapped or blank "
                    "image is indistinguishable from a working pipeline until "
                    "every AUC comes back at 0.50."
                ) from e
            size = DEFAULT_IMAGE_SIZE
            image = torch.zeros(1, size, size)

        return image, self.labels[idx], self.mask[idx], self.text_ids[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test CheXpertDataset on a small sample.")
    parser.add_argument('--parquet-path', required=True)
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--n-samples', type=int, default=32)
    args = parser.parse_args()

    from src.xray_report.dataloader import build_eval_transform

    df = pd.read_parquet(args.parquet_path).head(args.n_samples)
    vocab = load_vocab(args.vocab_path)

    ds = CheXpertDataset(
        df=df,
        image_root=args.image_root,
        token_to_idx=vocab['token_to_idx'],
        transform=build_eval_transform(),
        strict=False,
    )

    image, labels, mask, text_ids = ds[0]
    print(f"image:    {tuple(image.shape)} {image.dtype}")
    print(f"labels:   {tuple(labels.shape)}")
    print(f"mask:     {tuple(mask.shape)}")
    print(f"text_ids: {tuple(text_ids.shape)}")

    print(f"\nimage range: min={image.min():.1f} max={image.max():.1f}")
    print(f"expected:    min near -1024, max near 1024")

    print(f"\nmask coverage across {len(ds)} rows: "
          f"{ds.mask.mean().item():.3f} of entries contribute to the loss")
    pos_rate = (ds.labels * ds.mask).sum().item() / max(ds.mask.sum().item(), 1)
    print(f"positive rate among unmasked entries: {pos_rate:.3f}")