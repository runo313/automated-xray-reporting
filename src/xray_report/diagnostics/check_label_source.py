#!/usr/bin/env python3
"""
Are the CheXbert label JSON files aligned with their paths?

Your notebook's merge is keyed on path_to_image and is correct. Yet two views
of one study end up with different labels, which is visible in the raw merge
output. So the misalignment arrives from the label JSON itself, unless the
assumption is wrong that views of one study share labels.

This settles it using data you already have. CheXbert runs on report text. If
two images share the SAME report string but get different labels, the label
file is misaligned. That test needs no external ground truth.

Run on your laptop, where the source files live:

    python3 check_label_source.py \
        --csv ~/Downloads/df_chexpert_plus_240401.csv \
        --labels ~/Downloads/report_fixed.json \
                 ~/Downloads/findings_fixed.json
"""

import argparse
import json
import os

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


def agreement(df, group_key, label_cols, name):
    """
    Fraction of groups whose rows do not all share one label row.
    dropna=False so a NaN counts as a value, matching how the labels are used.
    """
    multi = group_key.duplicated(keep=False)
    if not multi.any():
        print(f"  {name}: no groups with more than one row")
        return None

    disagree = (
        df.loc[multi, label_cols]
        .groupby(group_key[multi])
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    frac = disagree.mean()
    print(f"  {name}: {len(disagree)} groups, "
          f"{int(disagree.sum())} disagree ({frac:.1%})")
    return frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True,
                    help="df_chexpert_plus_240401.csv")
    ap.add_argument('--labels', nargs='+', required=True,
                    help="one or more chexbert_labels/*.json files")
    ap.add_argument('--text-col', default='report',
                    help="column holding the text CheXbert was run on")
    args = ap.parse_args()

    print("loading CSV...")
    usecols = ['path_to_image', 'frontal_lateral', args.text_col]
    csv = pd.read_csv(args.csv, usecols=lambda c: c in usecols,
                      low_memory=False)
    print(f"  {len(csv)} rows\n")

    # ------------------------------------------------------ structure check
    print("=" * 66)
    print("STRUCTURE: does one study map to one report?")
    print("=" * 66)

    csv['_study'] = csv['path_to_image'].str.rsplit('/', n=1).str[0]
    per_study = csv.groupby('_study')[args.text_col].nunique(dropna=False)
    multi_report = (per_study > 1).sum()

    print(f"studies: {len(per_study)}")
    print(f"studies with more than one distinct {args.text_col}: "
          f"{multi_report} ({multi_report / len(per_study):.1%})")

    if multi_report / len(per_study) > 0.05:
        print("\n  Studies do NOT map cleanly to one report in this dataset,")
        print("  so cross-view label disagreement may be legitimate and the")
        print("  earlier finding needs reinterpreting. Read the report-text")
        print("  results below instead; they do not rely on that assumption.")
    else:
        print("\n  Confirmed: one study, one report. Views of a study must")
        print("  therefore share labels.")

    # ------------------------------------------------------ per label file
    for label_path in args.labels:
        name = os.path.basename(label_path)
        print()
        print("=" * 66)
        print(f"LABEL FILE: {name}")
        print("=" * 66)

        lab = load_jsonl(label_path)
        print(f"{len(lab)} rows, columns: {list(lab.columns)[:4]}...")

        if 'path_to_image' not in lab.columns:
            print("  no path_to_image column, skipping")
            continue

        present = [c for c in LABEL_COLS if c in lab.columns]
        print(f"label columns present: {len(present)}")

        dup = lab['path_to_image'].duplicated().sum()
        print(f"duplicate paths: {dup}\n")

        merged = lab.merge(csv, on='path_to_image', how='inner')
        print(f"joined rows: {len(merged)}\n")

        print("agreement within groups that MUST share labels:")
        agreement(merged, merged['_study'], present, "grouped by study")
        agreement(merged, merged[args.text_col], present,
                  f"grouped by identical {args.text_col} text")

        # A control: random groups of the same size should disagree a lot.
        shuffled = merged.sample(frac=1.0, random_state=0).reset_index(drop=True)
        fake_key = pd.Series(shuffled.index // 2, index=shuffled.index)
        print("\ncontrol (arbitrary pairs, disagreement should be HIGH):")
        agreement(shuffled, fake_key, present, "random pairs")

    print()
    print("=" * 66)
    print("HOW TO READ THIS")
    print("=" * 66)
    print("""
If 'grouped by identical report text' disagrees at a high rate, the label
file is misaligned against its own paths. No merge you write can fix that;
you need a different label source.

If it agrees but 'grouped by study' disagrees, then studies genuinely carry
more than one report in this release, the labels are fine, and the earlier
cross-view finding was my assumption being wrong rather than your data
being broken.

If both agree, the misalignment entered somewhere between these files and
the parquet, and the notebook needs another look.
""")


if __name__ == '__main__':
    main()