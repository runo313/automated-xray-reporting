#!/usr/bin/env python3
"""
Builds train/val/test DataLoaders.

Changed from the original:
  - ImageNet normalization replaced with xrv normalization. torchxrayvision
    models expect [-1024, 1024]; ImageNet normalization produces ~[-2, 2.6],
    which collapses the pretrained BatchNorm layers to constant output.
  - RandomHorizontalFlip removed. Laterality is clinically meaningful on chest
    radiographs; flipping teaches the model that left and right are the same.
  - Grayscale conversion moved here from the encoder, so the encoder can assert
    on channel count instead of silently averaging.
  - Normalization implemented as a class, not transforms.Lambda, so it pickles
    under num_workers > 0.

Place at: src/xray_report/dataloader.py
"""

import argparse

import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from src.xray_report.config import DEFAULT_IMAGE_SIZE, DEFAULT_MAX_LEN
from src.xray_report.datasets import CheXpertDataset
from src.xray_report.utils.vocabulary import load_vocab


class XRVNormalize:
    """
    Map a [0, 1] tensor from ToTensor into the [-1024, 1024] range that
    torchxrayvision models were trained on.

    Mirrors xrv.datasets.normalize(img, 255), which computes
    (2 * img / maxval - 1) * 1024.
    """

    def __call__(self, x):
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        return (2.0 * x - 1.0) * 1024.0

    def __repr__(self):
        return f"{self.__class__.__name__}(range=[-1024, 1024])"


def build_train_transform(image_size=DEFAULT_IMAGE_SIZE, augment=True):
    """Training transform. Augmentation is geometry-only and laterality-safe."""
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
    ]

    if augment:
        ops.append(
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=0,
            )
        )

    ops += [transforms.ToTensor(), XRVNormalize()]
    return transforms.Compose(ops)


def build_eval_transform(image_size=DEFAULT_IMAGE_SIZE):
    """Deterministic transform for val/test. No augmentation."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        XRVNormalize(),
    ])


def subsample(df, n, split_col='split', random_state=42):
    """
    Take roughly n training rows, keeping val and test intact.

    Sampling is at patient level when a patient column exists, so the same
    patient never appears in two splits after subsampling.
    """
    if n is None:
        return df

    train = df[df[split_col] == 'train']
    rest = df[df[split_col] != 'train']

    patient_col = next(
        (c for c in ('patient_id', 'patient', 'subject_id') if c in df.columns),
        None,
    )

    if patient_col:
        patients = train[patient_col].drop_duplicates()
        frac = min(1.0, n / max(len(train), 1))
        keep = patients.sample(frac=frac, random_state=random_state)
        train = train[train[patient_col].isin(keep)]
    else:
        train = train.sample(n=min(n, len(train)), random_state=random_state)

    return pd.concat([train, rest], ignore_index=True)


def build_dataloaders(df, image_root, token_to_idx, batch_size=32, num_workers=0,
                      max_len=DEFAULT_MAX_LEN, split_col='split',
                      image_size=DEFAULT_IMAGE_SIZE, augment=True,
                      train_subsample=None):
    """
    Build train/val/test DataLoaders from a single merged dataframe.

    Args:
        df: Full merged dataframe containing split_col with 'train'/'val'/'test'.
        image_root: Root folder containing downloaded images.
        token_to_idx: Vocabulary mapping from vocabulary.load_vocab.
        batch_size: Batch size for all three loaders.
        num_workers: DataLoader worker processes.
        max_len: Fixed length for encoded text sequences.
        split_col: Column indicating train/val/test membership.
        image_size: Square resize target. Must match encoder expectations (224).
        augment: Apply geometric augmentation to the training split.
        train_subsample: Approximate cap on training rows, for fast iteration.

    Returns:
        dict with keys 'train', 'val', 'test', each a DataLoader.
    """
    if train_subsample is not None:
        df = subsample(df, train_subsample, split_col=split_col)

    train_df = df[df[split_col] == 'train']
    val_df = df[df[split_col] == 'val']
    test_df = df[df[split_col] == 'test']

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            f"empty split: train={len(train_df)} val={len(val_df)} "
            f"test={len(test_df)}. Check '{split_col}' values."
        )

    train_tf = build_train_transform(image_size, augment=augment)
    eval_tf = build_eval_transform(image_size)

    train_ds = CheXpertDataset(train_df, image_root, token_to_idx, train_tf, max_len)
    val_ds = CheXpertDataset(val_df, image_root, token_to_idx, eval_tf, max_len)
    test_ds = CheXpertDataset(test_df, image_root, token_to_idx, eval_tf, max_len)

    common = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            drop_last=True, **common),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          drop_last=False, **common),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           drop_last=False, **common),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test train/val/test DataLoaders.")
    parser.add_argument('--parquet-path', required=True)
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path)
    vocab = load_vocab(args.vocab_path)

    loaders = build_dataloaders(df, args.image_root, vocab['token_to_idx'],
                                batch_size=args.batch_size)

    for split, loader in loaders.items():
        print(f"{split}: {len(loader.dataset)} examples, {len(loader)} batches")

    images, labels, masks, text_ids = next(iter(loaders['train']))
    print(f"\nbatch images:   {tuple(images.shape)}")
    print(f"batch labels:   {tuple(labels.shape)}")
    print(f"batch masks:    {tuple(masks.shape)}")
    print(f"batch text_ids: {tuple(text_ids.shape)}")

    # The check that would have caught the original bug in ten seconds.
    print(f"\nimage range: min={images.min():.1f} max={images.max():.1f} "
          f"mean={images.mean():.1f}")
    print("expected:    min near -1024, max near 1024")

    per_image_std = images.reshape(images.shape[0], -1).std(dim=1)
    print(f"per-image std: min={per_image_std.min():.1f} "
          f"max={per_image_std.max():.1f}   (near 0 means blank images)")

    assert images.shape[1] == 1, f"expected 1 channel, got {images.shape[1]}"
    assert images.abs().max() > 100, "input scale too small for torchxrayvision"