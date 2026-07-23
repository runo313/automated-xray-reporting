#!/usr/bin/env python3
"""Builds train/val/test DataLoaders for the CheXpert Plus pipeline."""

import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from src.xray_report.datasets import CheXpertDataset
from src.xray_report.utils.vocabulary import load_vocab
from src.xray_report.config import DEFAULT_IMAGE_SIZE, DEFAULT_MAX_LEN


def build_train_transform(image_size=DEFAULT_IMAGE_SIZE):
    """Resize/normalize transform with augmentation, for training only."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(image_size=DEFAULT_IMAGE_SIZE):
    """resize/normalize transform, for val/test."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_dataloaders(df, image_root, token_to_idx, batch_size=32, num_workers=0,
                       max_len=DEFAULT_MAX_LEN, split_col='split'):
    """
    Build train/val/test DataLoaders from a single merged dataframe.

    Args:
        df: Full merged dataframe containing a split_col with 'train'/'val'/'test'.
        image_root: Root folder containing downloaded images.
        token_to_idx: Vocabulary mapping loaded from vocabulary.load_vocab.
        batch_size: Batch size for all three loaders.
        num_workers: Number of DataLoader worker processes (0 for local, higher on EC2).
        max_len: Fixed length for encoded text sequences.
        split_col: Name of the column indicating train/val/test membership.

    Returns:
        dict with keys 'train', 'val', 'test', each a DataLoader.
    """
    train_df = df[df[split_col] == 'train']
    val_df = df[df[split_col] == 'val']
    test_df = df[df[split_col] == 'test']

    train_ds = CheXpertDataset(train_df, image_root, token_to_idx, build_train_transform(), max_len)
    val_ds = CheXpertDataset(val_df, image_root, token_to_idx, build_eval_transform(), max_len)
    test_ds = CheXpertDataset(test_df, image_root, token_to_idx, build_eval_transform(), max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test train/val/test DataLoaders.")
    parser.add_argument('--parquet-path', required=True, help="Path to merged labels parquet file")
    parser.add_argument('--vocab-path', required=True, help="Path to saved vocabulary pickle")
    parser.add_argument('--image-root', required=True, help="Root folder containing downloaded images")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size for the smoke test")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path)
    vocab = load_vocab(args.vocab_path)

    loaders = build_dataloaders(df, args.image_root, vocab['token_to_idx'], batch_size=args.batch_size)

    for split, loader in loaders.items():
        print(f"{split}: {len(loader.dataset)} examples, {len(loader)} batches")

    images, labels, masks, text_ids = next(iter(loaders['train']))
    print(f"batch images:   {images.shape}")
    print(f"batch labels:   {labels.shape}")
    print(f"batch masks:    {masks.shape}")
    print(f"batch text_ids: {text_ids.shape}")