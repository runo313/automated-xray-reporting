#!/usr/bin/env python3
"""
Pre-training sanity check. Run before spending another GPU hour.

Three gates. Stop at the first failure.

  1. Input range     - is the tensor in [-1024, 1024] like xrv expects?
  2. Feature variance- does the encoder produce different features per image?
  3. Zero-shot AUC   - can the pretrained CheXpert head classify untrained?

Gate 3 is decisive. The xrv "chex" weights were trained on CheXpert, so on a
CheXpert-derived val split they should score AUC ~0.75-0.85 on common findings
with zero fine-tuning. If they do, the data pipeline is sound and the previous
all-0.50 result was the normalization bug. If they don't, the problem is
upstream of training.

Usage: python3 sanity_check.py --parquet-path data/chexpert_plus_ready.parquet --image-root data/images --split val --n 500
Claude Opus 4.6
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

SPLIT_PREFIXES = ('train/', 'valid/', 'val/', 'test/')

# xrv pathology name -> CheXpert column name
NAME_MAP = {
    'Atelectasis': 'Atelectasis',
    'Cardiomegaly': 'Cardiomegaly',
    'Consolidation': 'Consolidation',
    'Edema': 'Edema',
    'Effusion': 'Pleural Effusion',
    'Lung Opacity': 'Lung Opacity',
    'Lung Lesion': 'Lung Lesion',
    'Pneumonia': 'Pneumonia',
    'Pneumothorax': 'Pneumothorax',
    'Enlarged Cardiomediastinum': 'Enlarged Cardiomediastinum',
    'Fracture': 'Fracture',
}


class XRVNormalize:
    """ToTensor gives [0,1]; xrv models expect [-1024, 1024], single channel."""

    def __call__(self, x):
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        return (2.0 * x - 1.0) * 1024.0


def build_transform(size=224):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        XRVNormalize(),
    ])


def strip_prefix(p):
    for prefix in SPLIT_PREFIXES:
        if p.startswith(prefix):
            return p[len(prefix):]
    return p


def load_batch(df, image_root, tf, idxs):
    """Return (tensor, kept_indices). Silently skips unreadable files."""
    imgs, kept = [], []
    for i in idxs:
        path = os.path.join(image_root, strip_prefix(df.at[i, 'path_to_image']))
        try:
            with Image.open(path) as im:
                imgs.append(tf(im))
            kept.append(i)
        except Exception:
            pass
    if not imgs:
        return None, []
    return torch.stack(imgs), kept


def gate_1(df, image_root, tf, n=32):
    print("=" * 64)
    print("GATE 1: input shape and range")
    print("=" * 64)

    batch, kept = load_batch(df, image_root, tf, list(df.index[:n]))
    if batch is None:
        print("  FAIL: could not load any images. Check --image-root.")
        return None

    print(f"loaded {len(kept)} of {n} probed")
    print(f"shape:  {tuple(batch.shape)}   (want (N, 1, 224, 224))")
    print(f"range:  min={batch.min():.1f}  max={batch.max():.1f}  "
          f"mean={batch.mean():.1f}")
    print("        expected min near -1024, max near 1024")

    stds = batch.reshape(batch.shape[0], -1).std(dim=1)
    print(f"per-image std: min={stds.min():.1f}  median={stds.median():.1f}")

    if batch.shape[1] != 1:
        print("\n  FAIL: input is not single-channel.")
        return None
    if batch.abs().max() < 100:
        print("\n  FAIL: input scale far below the xrv range. The pretrained")
        print("  BatchNorm running stats will collapse to constants.")
        return None
    if stds.min() < 1.0:
        print("\n  FAIL: at least one image is blank.")
        return None

    print("\n  PASS")
    return batch


def gate_2(model, batch, device):
    print()
    print("=" * 64)
    print("GATE 2: are encoder features image-dependent?")
    print("=" * 64)

    model.eval()
    with torch.no_grad():
        pooled = model.features(batch.to(device)).mean(dim=(2, 3))

    across = pooled.std(dim=0).mean().item()
    within = pooled.std(dim=1).mean().item()
    print(f"mean std ACROSS images (per feature): {across:.4f}")
    print(f"mean std WITHIN an image:            {within:.4f}")
    print("\nIf 'across' is near zero, every X-ray maps to the same vector and")
    print("no classifier head on top can beat base rates.")

    if across < 1e-3:
        print("\n  FAIL: features are constant across images.")
        return False

    print("\n  PASS")
    return True


def gate_3(model, df, image_root, tf, device, n, batch_size=32):
    print()
    print("=" * 64)
    print("GATE 3: zero-shot AUC, pretrained head, no training")
    print("=" * 64)

    sub = df.head(n).reset_index(drop=True)
    probs, rows = [], []

    model.eval()
    for start in range(0, len(sub), batch_size):
        idxs = list(sub.index[start:start + batch_size])
        batch, kept = load_batch(sub, image_root, tf, idxs)
        if batch is None:
            continue
        with torch.no_grad():
            out = torch.sigmoid(model(batch.to(device)))
        probs.append(out.cpu().numpy())
        rows.extend(kept)

    probs = np.concatenate(probs)
    sub = sub.loc[rows].reset_index(drop=True)
    print(f"scored {len(sub)} images\n")

    xrv_names = list(model.pathologies)
    good = 0

    for xrv_name, col in NAME_MAP.items():
        if xrv_name not in xrv_names or col not in sub.columns:
            continue
        y = pd.to_numeric(sub[col], errors='coerce').values
        valid = np.isin(y, [0.0, 1.0])
        if valid.sum() < 20 or len(np.unique(y[valid])) < 2:
            continue
        auc = roc_auc_score(y[valid], probs[valid, xrv_names.index(xrv_name)])
        flag = "ok" if auc > 0.70 else "LOW"
        good += auc > 0.70
        print(f"  {col:<30} AUC={auc:.3f}  n={int(valid.sum()):<5} {flag}")

    print()
    if good >= 3:
        print("  PASS - pipeline and weights are sound. The previous all-0.50")
        print("  result was the ImageNet-normalization bug. Proceed to training.")
        return True

    print("  FAIL - the pretrained CheXpert head cannot classify your own")
    print("  CheXpert data. Something upstream of training is still wrong.")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet-path', default='data/chexpert_plus_ready.parquet')
    ap.add_argument('--image-root', default='data/images')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--image-size', type=int, default=224)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    print(f"torch:  {torch.__version__}\n")

    df = pd.read_parquet(args.parquet_path)
    if 'split' in df.columns and args.split != 'all':
        df = df[df['split'] == args.split]
    df = df.reset_index(drop=True)
    print(f"{len(df)} rows in split='{args.split}'\n")

    tf = build_transform(args.image_size)

    batch = gate_1(df, args.image_root, tf)
    if batch is None:
        return

    model = xrv.models.DenseNet(weights="densenet121-res224-chex").to(device)

    if not gate_2(model, batch, device):
        return

    gate_3(model, df, args.image_root, tf, device, args.n)


if __name__ == '__main__':
    main()