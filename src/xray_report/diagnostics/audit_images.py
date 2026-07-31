#!/usr/bin/env python3
"""
Why does a CheXpert-pretrained model fail on CheXpert data?

Preprocessing is confirmed correct and features vary across images, yet
zero-shot AUC is at or below chance. Two hypotheses remain:

  A. The images are photometrically inverted (lungs bright, bone dark).
     Signature: AUCs consistently BELOW 0.5, and re-testing with inverted
     input flips them above 0.5.

  B. The parquet's labels do not correspond to the images they sit beside.
     Signature: internal label logic is violated, or two views of the same
     study carry different labels.

This runs both tests plus a visual export.

Usage (from the project root):
    python3 diagnose.py
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

LABEL_COLS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices', 'No Finding',
]

NAME_MAP = {
    'Atelectasis': 'Atelectasis',
    'Cardiomegaly': 'Cardiomegaly',
    'Consolidation': 'Consolidation',
    'Edema': 'Edema',
    'Effusion': 'Pleural Effusion',
    'Lung Opacity': 'Lung Opacity',
    'Pneumothorax': 'Pneumothorax',
    'Enlarged Cardiomediastinum': 'Enlarged Cardiomediastinum',
}


def strip_prefix(p):
    for prefix in SPLIT_PREFIXES:
        if p.startswith(prefix):
            return p[len(prefix):]
    return p


def to_tensor(img, size=224, invert=False):
    """PIL image -> (1, size, size) tensor in the xrv [-1024, 1024] range."""
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    x = tf(img)
    if invert:
        x = 1.0 - x
    return (2.0 * x - 1.0) * 1024.0


# ------------------------------------------------------------------ test 0

def raw_properties(df, image_root, n=8):
    """What are these PNGs, actually?"""
    print("=" * 66)
    print("0. RAW FILE PROPERTIES")
    print("=" * 66)

    for rel in df['path_to_image'].head(n):
        path = os.path.join(image_root, strip_prefix(rel))
        try:
            with Image.open(path) as im:
                arr = np.array(im.convert('L'))
                print(f"{os.path.basename(rel):<26} mode={im.mode:<6} "
                      f"size={str(im.size):<12} "
                      f"8bit[min={arr.min():>3} max={arr.max():>3} "
                      f"mean={arr.mean():>6.1f} std={arr.std():>5.1f}]")
        except Exception as e:
            print(f"{rel}: {e}")

    print("\nA 16-bit source ('I;16') converted carelessly can clip or invert.")
    print("size should be 256x256 from the download step.\n")


# ------------------------------------------------------------------ test A

def inversion_heuristic(df, image_root, n=64):
    """
    In a standard chest radiograph, air is dark and bone is bright, so the
    lung fields (mid-left and mid-right) are darker than the mediastinum
    (centre column). If that relationship is reversed, the image is inverted.
    """
    print("=" * 66)
    print("A1. PHOTOMETRIC ORIENTATION HEURISTIC")
    print("=" * 66)

    lung_darker = 0
    total = 0

    for rel in df['path_to_image'].head(n):
        path = os.path.join(image_root, strip_prefix(rel))
        try:
            with Image.open(path) as im:
                a = np.array(im.convert('L').resize((224, 224)), dtype=np.float32)
        except Exception:
            continue

        mid = slice(70, 150)
        left_lung = a[mid, 40:85].mean()
        right_lung = a[mid, 139:184].mean()
        mediastinum = a[mid, 100:124].mean()

        total += 1
        if (left_lung + right_lung) / 2 < mediastinum:
            lung_darker += 1

    frac = lung_darker / max(total, 1)
    print(f"lung fields darker than mediastinum in {lung_darker}/{total} "
          f"({frac:.1%})")
    print("\nNormal radiographs: expect well above 80%.")
    print("Below ~30% means the images are photometrically inverted.")
    if 0.3 <= frac <= 0.8:
        print("A middling value is inconclusive; rely on the visual export.")
    print()
    return frac


def zero_shot(model, df, image_root, device, n, invert, batch_size=32):
    """Zero-shot AUC, optionally with inverted input."""
    sub = df.head(n).reset_index(drop=True)
    probs, rows = [], []

    model.eval()
    for start in range(0, len(sub), batch_size):
        imgs, kept = [], []
        for i in sub.index[start:start + batch_size]:
            path = os.path.join(image_root, strip_prefix(sub.at[i, 'path_to_image']))
            try:
                with Image.open(path) as im:
                    imgs.append(to_tensor(im, invert=invert))
                kept.append(i)
            except Exception:
                pass
        if not imgs:
            continue
        with torch.no_grad():
            out = model(torch.stack(imgs).to(device))
        probs.append(out.cpu().numpy())
        rows.extend(kept)

    probs = np.concatenate(probs)
    sub = sub.loc[rows].reset_index(drop=True)

    xrv_names = list(model.pathologies)
    results = {}
    for xrv_name, col in NAME_MAP.items():
        if xrv_name not in xrv_names or col not in sub.columns:
            continue
        y = pd.to_numeric(sub[col], errors='coerce').values
        valid = np.isin(y, [0.0, 1.0])
        if valid.sum() < 20 or len(np.unique(y[valid])) < 2:
            continue
        col_idx = xrv_names.index(xrv_name)
        p = probs[valid, col_idx]
        if np.isnan(p).any():
            results[col] = None
            continue
        results[col] = roc_auc_score(y[valid], p)
    return results


def inversion_ab_test(model, df, image_root, device, n=500):
    print("=" * 66)
    print("A2. A/B TEST: NORMAL vs INVERTED INPUT")
    print("=" * 66)

    normal = zero_shot(model, df, image_root, device, n, invert=False)
    flipped = zero_shot(model, df, image_root, device, n, invert=True)

    print(f"{'label':<30} {'normal':>8} {'inverted':>9}")
    print("-" * 49)
    better = 0
    for col in normal:
        a, b = normal[col], flipped.get(col)
        if a is None or b is None:
            continue
        mark = "  <-- inverted wins" if b > a + 0.05 else ""
        better += b > a + 0.05
        print(f"{col:<30} {a:>8.3f} {b:>9.3f}{mark}")

    print()
    if better >= 3:
        print("  CONFIRMED: the images are photometrically inverted. Invert them")
        print("  at load time (1.0 - x after ToTensor) and everything downstream")
        print("  should work.")
    else:
        print("  Inversion is NOT the explanation. Move to the label tests.")
    print()
    return better >= 3


# ------------------------------------------------------------------ test B

def label_consistency(clean_parquet):
    """
    CheXpert labels have internal structure. If it is violated, the label
    matrix does not line up with the rows it sits in.
    """
    print("=" * 66)
    print("B1. INTERNAL LABEL CONSISTENCY")
    print("=" * 66)

    df = pd.read_parquet(clean_parquet)
    present = [c for c in LABEL_COLS if c in df.columns]
    vals = df[present].apply(pd.to_numeric, errors='coerce')

    # 'No Finding' == 1 should imply every other finding is not positive.
    if 'No Finding' in vals.columns:
        nf = vals['No Finding'] == 1.0
        others = [c for c in present if c not in ('No Finding', 'Support Devices')]
        conflict = (vals.loc[nf, others] == 1.0).any(axis=1)
        print(f"rows with No Finding == 1:              {int(nf.sum())}")
        print(f"  of those, also positive elsewhere:    {int(conflict.sum())} "
              f"({conflict.mean():.1%})")
        print("  expected: close to 0%\n")

    # Two views of one study share a label row in CheXpert.
    if 'path_to_image' in df.columns:
        study = df['path_to_image'].str.rsplit('/', n=1).str[0]
        multi = study.duplicated(keep=False)
        if multi.any():
            grouped = vals[multi].groupby(study[multi])
            disagree = grouped.nunique(dropna=False).gt(1).any(axis=1)
            print(f"studies with more than one view:        {len(disagree)}")
            print(f"  where the views disagree on labels:   {int(disagree.sum())} "
                  f"({disagree.mean():.1%})")
            print("  expected: 0%. Anything else means rows are scrambled.\n")
        else:
            print("no multi-view studies found (frontal-only parquet?)\n")


# ------------------------------------------------------------------ visual

def export_montage(df, image_root, out_path, rows=3, cols=4, tile=224):
    """Write a labelled grid so the images can actually be looked at."""
    print("=" * 66)
    print("C. VISUAL EXPORT")
    print("=" * 66)

    canvas = Image.new('L', (cols * tile, rows * tile), color=0)
    picked = []

    for k, rel in enumerate(df['path_to_image'].head(rows * cols)):
        path = os.path.join(image_root, strip_prefix(rel))
        try:
            with Image.open(path) as im:
                canvas.paste(im.convert('L').resize((tile, tile)),
                             ((k % cols) * tile, (k // cols) * tile))
            picked.append(rel)
        except Exception:
            pass

    canvas.save(out_path)
    print(f"wrote {out_path}")
    print("\nDownload it and look. You are checking:")
    print("  - lungs DARK, spine and ribs BRIGHT  (correct polarity)")
    print("  - lungs bright, bone dark            (inverted)")
    print("  - recognisable chest radiographs at all")
    print("  - reasonable proportions, not obviously squashed\n")

    for rel in picked[:4]:
        print(f"  {rel}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet-path', default='data/chexpert_plus_ready.parquet')
    ap.add_argument('--clean-parquet', default='data/chexpert_plus_clean.parquet')
    ap.add_argument('--image-root', default='data/images')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--montage', default='diagnostic_montage.png')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_parquet(args.parquet_path)
    if 'split' in df.columns:
        df = df[df['split'] == args.split]
    df = df.reset_index(drop=True)
    print(f"device: {device}   rows in '{args.split}': {len(df)}\n")

    raw_properties(df, args.image_root)
    inversion_heuristic(df, args.image_root)

    model = xrv.models.DenseNet(weights="densenet121-res224-chex").to(device)
    inverted = inversion_ab_test(model, df, args.image_root, device, args.n)

    if not inverted and os.path.exists(args.clean_parquet):
        label_consistency(args.clean_parquet)

    export_montage(df, args.image_root, args.montage)


if __name__ == '__main__':
    main()
