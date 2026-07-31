#!/usr/bin/env python3
"""
Are the IMAGES intact, or were they scrambled too?

B1 showed labels disagree across views of the same study, which means labels
are not paired with the right images. Before rebuilding the parquet, confirm
the images themselves are fine. Every test here is label-free.

  1. Are the model's predictions degenerate (same score for every image)?
  2. Does the model produce clinically coherent structure? Cardiomegaly and
     Enlarged Cardiomediastinum should correlate strongly; Cardiomegaly and
     Fracture should not.
  3. Do two views of the SAME study get similar predictions? A frontal and a
     lateral of one patient share anatomy, so study-level findings like
     Cardiomegaly should agree far more within a study than across random
     pairs. If they do not, the image files are scrambled against their paths.

Test 3 is the decisive one. If it passes, the images are fine and rebuilding
the label pairing fixes everything.

Usage (from the project root):
    python3 check_images.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from PIL import Image
from torchvision import transforms

SPLIT_PREFIXES = ('train/', 'valid/', 'val/', 'test/')

# Clinically linked in xrv's output space. A working model on good images
# shows clear positive correlation on these.
RELATED = [
    ('Cardiomegaly', 'Enlarged Cardiomediastinum'),
    ('Effusion', 'Edema'),
    ('Lung Opacity', 'Consolidation'),
    ('Consolidation', 'Pneumonia'),
]

# Not clinically linked. Correlation here should be clearly weaker.
UNRELATED = [
    ('Cardiomegaly', 'Pneumothorax'),
    ('Fracture', 'Edema'),
]


def strip_prefix(p):
    for prefix in SPLIT_PREFIXES:
        if p.startswith(prefix):
            return p[len(prefix):]
    return p


def build_transform(size=224):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])


def score(model, paths, image_root, tf, device, batch_size=32):
    """Return (probs, kept_paths). Raw sigmoid scores, no op_norm."""
    probs, kept = [], []

    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        imgs, ok = [], []
        for rel in chunk:
            full = os.path.join(image_root, strip_prefix(rel))
            try:
                with Image.open(full) as im:
                    x = tf(im)
                imgs.append((2.0 * x - 1.0) * 1024.0)
                ok.append(rel)
            except Exception:
                pass
        if not imgs:
            continue
        with torch.no_grad():
            out = torch.sigmoid(model(torch.stack(imgs).to(device)))
        probs.append(out.cpu().numpy())
        kept.extend(ok)

    return np.concatenate(probs), kept


def test_1_degenerate(probs, names):
    print("=" * 66)
    print("1. ARE PREDICTIONS DEGENERATE?")
    print("=" * 66)

    stds = probs.std(axis=0)
    interesting = [(names[i], probs[:, i].mean(), stds[i])
                   for i in range(len(names)) if names[i]]

    print(f"{'pathology':<30} {'mean':>7} {'std':>7}")
    print("-" * 46)
    for name, m, s in sorted(interesting, key=lambda t: -t[2])[:10]:
        print(f"{name:<30} {m:>7.3f} {s:>7.4f}")

    worst = max(s for _, _, s in interesting)
    print(f"\nlargest per-pathology std: {worst:.4f}")
    if worst < 0.01:
        print("  FAIL: the model outputs nearly the same score for every image.")
        return False
    print("  PASS: predictions vary across images.\n")
    return True


def test_2_structure(probs, names):
    print("=" * 66)
    print("2. IS THERE CLINICALLY COHERENT STRUCTURE?")
    print("=" * 66)

    idx = {n: i for i, n in enumerate(names) if n}

    def corr(a, b):
        if a not in idx or b not in idx:
            return None
        return float(np.corrcoef(probs[:, idx[a]], probs[:, idx[b]])[0, 1])

    rel_vals, unrel_vals = [], []

    print("related pairs (expect strong positive correlation):")
    for a, b in RELATED:
        r = corr(a, b)
        if r is not None:
            rel_vals.append(r)
            print(f"  {a:<28} vs {b:<28} r={r:+.3f}")

    print("\nunrelated pairs (expect weaker):")
    for a, b in UNRELATED:
        r = corr(a, b)
        if r is not None:
            unrel_vals.append(r)
            print(f"  {a:<28} vs {b:<28} r={r:+.3f}")

    if not rel_vals:
        print("\n  could not compute; pathology names missing")
        return False

    gap = np.mean(rel_vals) - np.mean(unrel_vals or [0.0])
    print(f"\nmean related {np.mean(rel_vals):+.3f}  "
          f"mean unrelated {np.mean(unrel_vals or [0.0]):+.3f}  "
          f"gap {gap:+.3f}")

    if gap > 0.15:
        print("  PASS: the model is reading real anatomy, not noise.\n")
        return True
    print("  WEAK: little coherent structure. The images may be degraded.\n")
    return False


def test_3_cross_view(model, clean_parquet, image_root, tf, device,
                      names, n_studies=150):
    """
    Two views of one study share a patient and a chest. If the files on disk
    match their paths, a frontal and a lateral from one study should get more
    similar predictions than two images picked at random.
    """
    print("=" * 66)
    print("3. DO TWO VIEWS OF ONE STUDY AGREE? (decisive)")
    print("=" * 66)

    df = pd.read_parquet(clean_parquet)
    study = df['path_to_image'].str.rsplit('/', n=1).str[0]

    pairs = []
    for _, grp in df.groupby(study):
        if len(grp) < 2:
            continue
        pairs.append((grp['path_to_image'].iloc[0], grp['path_to_image'].iloc[1]))
        if len(pairs) >= n_studies:
            break

    if not pairs:
        print("  no multi-view studies found; cannot run this test\n")
        return None

    first = [a for a, _ in pairs]
    second = [b for _, b in pairs]

    p1, k1 = score(model, first, image_root, tf, device)
    p2, k2 = score(model, second, image_root, tf, device)
    m = min(len(k1), len(k2))
    p1, p2 = p1[:m], p2[:m]
    print(f"scored {m} study pairs\n")

    idx = {n: i for i, n in enumerate(names) if n}
    targets = [t for t in ('Cardiomegaly', 'Effusion', 'Edema',
                           'Enlarged Cardiomediastinum') if t in idx]

    rng = np.random.default_rng(0)
    shuffled = rng.permutation(m)

    print(f"{'pathology':<30} {'same study':>11} {'random pair':>12}")
    print("-" * 55)

    wins = 0
    for t in targets:
        i = idx[t]
        real = float(np.corrcoef(p1[:, i], p2[:, i])[0, 1])
        fake = float(np.corrcoef(p1[:, i], p2[shuffled, i])[0, 1])
        wins += real > fake + 0.15
        print(f"{t:<30} {real:>+11.3f} {fake:>+12.3f}")

    print()
    if wins >= 2:
        print("  PASS: images match their paths. The files on disk are correct")
        print("  and the problem is purely the label pairing in the parquet.")
        return True

    print("  FAIL: two views of one study look unrelated to the model. The")
    print("  image files may be scrambled against their filenames too, which")
    print("  rebuilding the parquet would not fix.")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet-path', default='data/chexpert_plus_ready.parquet')
    ap.add_argument('--clean-parquet', default='data/chexpert_plus_clean.parquet')
    ap.add_argument('--image-root', default='data/images')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=400)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = build_transform()

    model = xrv.models.DenseNet(weights="densenet121-res224-chex").to(device)
    model.eval()
    model.op_threshs = None      # raw logits, avoids NaN from operating-point norm
    names = list(model.pathologies)

    df = pd.read_parquet(args.parquet_path)
    if 'split' in df.columns:
        df = df[df['split'] == args.split]
    paths = df['path_to_image'].head(args.n).tolist()

    print(f"device: {device}   scoring {len(paths)} images\n")
    probs, kept = score(model, paths, args.image_root, tf, device)
    print(f"scored {len(kept)}\n")

    test_1_degenerate(probs, names)
    test_2_structure(probs, names)

    if os.path.exists(args.clean_parquet):
        test_3_cross_view(model, args.clean_parquet, args.image_root,
                          tf, device, names)
    else:
        print(f"skipping test 3: {args.clean_parquet} not found")


if __name__ == '__main__':
    main()
