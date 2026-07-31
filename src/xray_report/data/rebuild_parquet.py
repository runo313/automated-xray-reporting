#!/usr/bin/env python3
"""
Rebuild the training parquet from a CheXbert label file that is actually
aligned with its paths.

report_fixed.json is misaligned: within-study label disagreement is 92.4%
against a 99.8% random baseline, meaning no within-study structure at all.
findings_fixed.json is clean at 0.3%. Check any candidate before trusting it.

This script validates first and refuses to write if the label file fails.

Screen every label file you have:
    python3 rebuild_from_chexbert.py --csv ~/Downloads/df_chexpert_plus_240401.csv \
        --screen ~/Downloads/*.json

Then build from whichever passed:
    python3 rebuild_from_chexbert.py --csv ~/Downloads/df_chexpert_plus_240401.csv \
        --label-json ~/Downloads/impression_fixed.json \
        --text-col section_impression \
        --out chexpert_plus_fixed.parquet
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

LABEL_COLS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices', 'No Finding',
]


def load_jsonl(path):
    with open(path) as f:
        return pd.DataFrame([json.loads(line) for line in f])


def study_of(paths):
    return paths.str.rsplit('/', n=1).str[0]


def alignment_score(lab, label_cols):
    """
    Returns (within_study_disagreement, random_pair_disagreement).

    A well-aligned file shows within-study disagreement far below the random
    baseline. A misaligned one sits right on top of it.
    """
    study = study_of(lab['path_to_image'])
    multi = study.duplicated(keep=False)
    if not multi.any():
        return None, None

    within = (
        lab.loc[multi, label_cols]
        .groupby(study[multi])
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
        .mean()
    )

    shuffled = lab.sample(frac=1.0, random_state=0).reset_index(drop=True)
    fake = pd.Series(shuffled.index // 2, index=shuffled.index)
    control = (
        shuffled[label_cols]
        .groupby(fake)
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
        .mean()
    )

    return float(within), float(control)


def screen(paths, label_cols=None):
    print("=" * 70)
    print("SCREENING LABEL FILES")
    print("=" * 70)
    print(f"{'file':<34} {'within-study':>13} {'random':>9} {'verdict':>10}")
    print("-" * 70)

    passed = []
    for p in paths:
        try:
            lab = load_jsonl(p)
        except Exception as e:
            print(f"{os.path.basename(p):<34} could not read: {e}")
            continue

        if 'path_to_image' not in lab.columns:
            continue
        cols = [c for c in (label_cols or LABEL_COLS) if c in lab.columns]
        if not cols:
            continue

        within, control = alignment_score(lab, cols)
        if within is None:
            continue

        # Aligned means clearly structured relative to the random baseline.
        ok = within < 0.10 and within < control * 0.5
        if ok:
            passed.append(p)
        print(f"{os.path.basename(p):<34} {within:>12.1%} {control:>8.1%} "
              f"{'ALIGNED' if ok else 'MISALIGNED':>10}")

    print()
    if passed:
        print("usable label files:")
        for p in passed:
            print(f"  {p}")
    else:
        print("no label file passed. Fall back to the original CheXpert v1.0")
        print("train.csv, which joins on the same path key and carries dense,")
        print("well-validated labels.")
    print()
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--screen', nargs='*', default=None,
                    help="screen these label files and exit")
    ap.add_argument('--label-json', default=None)
    ap.add_argument('--out', default='chexpert_plus_fixed.parquet')
    ap.add_argument('--text-col', default='section_impression')
    ap.add_argument('--frontal-only', action='store_true', default=True)
    ap.add_argument('--keep-laterals', dest='frontal_only', action='store_false')
    ap.add_argument('--require-labels', action='store_true', default=True,
                    help="drop rows where every label is blank")
    ap.add_argument('--image-root', default=None,
                    help="if set, drop rows whose image is not on disk")
    ap.add_argument('--val-frac', type=float, default=0.05)
    ap.add_argument('--test-frac', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if args.screen:
        screen(args.screen)
        return

    if not args.label_json:
        raise SystemExit("pass --label-json, or --screen to evaluate candidates")

    # ------------------------------------------------------------- validate
    lab = load_jsonl(args.label_json)
    cols = [c for c in LABEL_COLS if c in lab.columns]
    print(f"label file: {os.path.basename(args.label_json)}")
    print(f"  {len(lab)} rows, {len(cols)} label columns, "
          f"{lab['path_to_image'].duplicated().sum()} duplicate paths")

    within, control = alignment_score(lab, cols)
    print(f"  within-study disagreement: {within:.1%} "
          f"(random baseline {control:.1%})")

    if not (within < 0.10 and within < control * 0.5):
        raise SystemExit(
            "\nThis label file is misaligned with its own paths. Building from\n"
            "it would reproduce the original bug. Screen your other files with\n"
            "  --screen ~/Downloads/*.json\n"
            "or fall back to the original CheXpert v1.0 train.csv."
        )
    print("  ALIGNED, proceeding\n")

    # ------------------------------------------------------------- merge
    keep_csv = ['path_to_image', 'frontal_lateral', 'split', args.text_col]
    csv = pd.read_csv(args.csv, usecols=lambda c: c in keep_csv,
                      low_memory=False)
    print(f"CSV: {len(csv)} rows")

    df = lab.merge(csv, on='path_to_image', how='inner', validate='one_to_one')
    print(f"joined: {len(df)} rows\n")

    # ------------------------------------------------------------- filter
    print("filtering")

    if args.require_labels:
        vals = df[cols].apply(pd.to_numeric, errors='coerce')
        has_any = vals.notna().any(axis=1)
        print(f"  rows with at least one label: {int(has_any.sum())} "
              f"({has_any.mean():.1%})")
        df = df[has_any].copy()

    if args.text_col in df.columns:
        has_text = df[args.text_col].notna()
        print(f"  rows with {args.text_col}: {int(has_text.sum())} "
              f"({has_text.mean():.1%})")
        df = df[has_text].copy()

    if args.frontal_only:
        if 'frontal_lateral' in df.columns:
            keep = df['frontal_lateral'].astype(str).str.lower().eq('frontal')
        else:
            keep = ~df['path_to_image'].str.contains('lateral', case=False)
        before = len(df)
        df = df[keep].copy()
        print(f"  dropped {before - len(df)} laterals")

    # Images on disk are PNG; the CSV lists JPG.
    df['path_to_image'] = df['path_to_image'].str.replace('.jpg', '.png',
                                                          regex=False)

    if args.image_root:
        full = df['path_to_image'].map(
            lambda p: os.path.join(args.image_root,
                                   p.split('/', 1)[1] if '/' in p else p)
        )
        on_disk = full.map(os.path.exists)
        print(f"  on disk: {int(on_disk.sum())}, missing: {int((~on_disk).sum())}")
        df = df[on_disk].copy()

    print(f"  remaining: {len(df)}\n")

    # ------------------------------------------------------------- splits
    df['_patient'] = df['path_to_image'].str.extract(r'(patient\d+)',
                                                     expand=False)
    patients = df['_patient'].dropna().unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(patients)

    n = len(patients)
    n_test = int(args.test_frac * n)
    n_val = int(args.val_frac * n)

    assign = {}
    assign.update({p: 'test' for p in patients[:n_test]})
    assign.update({p: 'val' for p in patients[n_test:n_test + n_val]})
    assign.update({p: 'train' for p in patients[n_test + n_val:]})
    df['split'] = df['_patient'].map(assign)

    print("splits (patient-level)")
    print(df['split'].value_counts().to_string())

    groups = {s: set(g['_patient']) for s, g in df.groupby('split')}
    names = list(groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if groups[names[i]] & groups[names[j]]:
                raise SystemExit(f"leakage: {names[i]} / {names[j]}")
    print("  no patient in more than one split\n")

    # ------------------------------------------------------------- report
    rows = []
    train = df['split'] == 'train'
    for c in cols:
        v = pd.to_numeric(df[c], errors='coerce')
        rows.append({
            'label': c,
            'prevalence': float((v[train] == 1.0).mean()),
            'n_pos_train': int((v[train] == 1.0).sum()),
            'n_uncertain': int((v == -1.0).sum()),
            'n_blank': int(v.isna().sum()),
        })
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda x: f"{x:.4f}"))

    out = df[['path_to_image', 'split'] + cols +
             [c for c in (args.text_col,) if c in df.columns]]
    out.reset_index(drop=True).to_parquet(args.out, index=False)
    print(f"\nwrote {len(out)} rows to {args.out}")
    print("\nUpload it and re-run sanity_check.py. Zero-shot AUC on")
    print("Cardiomegaly, Edema and Pleural Effusion should now be well")
    print("above chance.")


if __name__ == '__main__':
    main()