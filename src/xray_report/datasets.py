#!/usr/bin/env python3
import argparse
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from src.xray_report.utils.vocabulary import tokenize, encode, load_vocab
from src.xray_report.config import LABEL_COLS, DEFAULT_MAX_LEN, DEFAULT_IMAGE_SIZE


class CheXpertDataset(Dataset):
    def __init__(self,df, image_root, token_to_idx, transform=None, max_len=DEFAULT_MAX_LEN):

        """
        Args:
            df: Dataframe with path_to_image, label columns, and section_impression.
            image_root: Root folder containing the image files.
            token_to_idx: Vocabulary mapping from vocabulary.build_vocab / load_vocab.
            transform: torchvision transform pipeline applied to each image.
            max_len: Fixed length to pad/truncate encoded text to.
        """
        self.df=df.reset_index(drop=True)
        self.image_root=image_root
        self.token_to_idx = token_to_idx
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Return one training example as (image, labels, mask, text_ids)."""
        row = self.df.iloc[idx]
        raw_path = row['path_to_image']
        clean_path = raw_path.replace('train/', '', 1).replace('valid/', '', 1)
        image_path = f"{self.image_root}/{clean_path}"
        image= Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels= row[LABEL_COLS].fillna(0.0)
        mask=(labels != -1.0).astype(float)
        labels= labels.replace(-1.0, 0.0)
        text_ids = encode(tokenize(row['section_impression']), self.token_to_idx, self.max_len)
        return (image,
                torch.tensor(labels.values.astype(float), dtype=torch.float32),
                torch.tensor(mask.values.astype(float), dtype=torch.float32),
                torch.tensor(text_ids, dtype=torch.long))

def build_default_transform(image_size=DEFAULT_IMAGE_SIZE):
    """Standard ImageNet-normalized resize transform for a pretrained CNN encoder."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test CheXpertDataset on a small sample.")
    parser.add_argument('--parquet-path', required=True, help="Path to merged labels parquet file")
    parser.add_argument('--vocab-path', required=True, help="Path to saved vocabulary pickle")
    parser.add_argument('--image-root', required=True, help="Root folder containing downloaded images")
    parser.add_argument('--n-samples', type=int, default=3, help="Number of rows to test on")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path).head(args.n_samples)
    vocab = load_vocab(args.vocab_path)

    ds = CheXpertDataset(
        df=df,
        image_root=args.image_root,
        token_to_idx=vocab['token_to_idx'],
        transform=build_default_transform(),
    )
    image, labels, mask, text_ids = ds[0]
    print(f"image: {image.shape} {image.dtype}")
    print(f"labels: {labels.shape}")
    print(f"mask: {mask.shape}")
    print(f"text_ids: {text_ids.shape}")