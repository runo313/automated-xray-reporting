#!/usr/bin/env python3
"""
Rule-based CheXpert-style labeler for radiology text, plus a self-validation
harness.

Why not run CheXbert itself: it needs a Stanford agreement, a 500MB
checkpoint, and a dependency chain. Why this is defensible instead: the labels
already in the parquet ARE CheXbert's output on these exact impressions, so
agreement between this labeler and those labels can be measured and quoted.
Validate first, then use it to score generated text.

    # how well does it reproduce CheXbert on reference text?
    python3 -m src.xray_report.eval.text_labeler --validate --n 5000

    # label arbitrary text
    python3 -m src.xray_report.eval.text_labeler \
        --text "no focal consolidation or pleural effusion."

Changes from the first version, driven by a 0.794 macro F1 run in which
precision was 0.93-1.00 on every label while recall lagged:
  - CheXpert's label hierarchy is now propagated. Consolidation, Edema,
    Pneumonia, Atelectasis and Lung Lesion imply Lung Opacity; Cardiomegaly
    implies Enlarged Cardiomediastinum. This was the single biggest gap
    (Enlarged Cardiomediastinum recall 0.155, Lung Opacity recall 0.665,
    both at near-perfect precision).
  - No Finding now fires whenever nothing else is positive, matching
    CheXbert, instead of requiring an explicit normal phrase.
  - Vocabulary expanded for Enlarged Cardiomediastinum, Pneumonia and
    Support Devices, the three weakest by recall.

Place at: src/xray_report/eval/text_labeler.py
"""

import argparse
import re

import numpy as np
import pandas as pd

LABEL_COLS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices', 'No Finding',
]

# CheXpert's label hierarchy. A positive child implies a positive parent, and
# the CheXbert labeler propagates that way. Not modelling this was costing
# most of the recall on the two parent labels.
PARENT_OF = {
    'Consolidation': 'Lung Opacity',
    'Edema': 'Lung Opacity',
    'Pneumonia': 'Lung Opacity',
    'Atelectasis': 'Lung Opacity',
    'Lung Lesion': 'Lung Opacity',
    'Cardiomegaly': 'Enlarged Cardiomediastinum',
}

# Surface forms per finding. Matched longest-first so that 'pleural effusion'
# wins over a bare 'effusion' when both could fire.
PHRASES = {
    'Enlarged Cardiomediastinum': [
        'enlarged cardiomediastinal', 'cardiomediastinal enlargement',
        'widened mediastinum', 'mediastinal widening',
        'widening of the mediastinum', 'enlarged mediastinal',
        'mediastinal enlargement', 'cardiomediastinal silhouette is enlarged',
        'prominent cardiomediastinal', 'widened cardiomediastinal',
        'mediastinal contour', 'mediastinal mass', 'aortic tortuosity',
        'tortuous aorta', 'ectatic aorta', 'unfolded aorta',
        'prominent mediastinum', 'mediastinal shift',
    ],
    'Cardiomegaly': [
        'cardiomegaly', 'enlarged heart', 'heart is enlarged',
        'cardiac enlargement', 'enlarged cardiac silhouette',
        'enlargement of the cardiac silhouette', 'enlarged heart size',
        'increased heart size', 'cardiac silhouette is enlarged',
        'enlarged cardiac contour', 'heart size is enlarged',
    ],
    'Lung Opacity': [
        'opacity', 'opacities', 'opacification', 'infiltrate', 'infiltrates',
        'airspace opacity', 'increased density', 'densities', 'reticular',
        'interstitial markings', 'interstitial prominence',
        'reticulonodular', 'ground glass', 'ground-glass', 'haziness',
        'increased marking', 'increased markings',
    ],
    'Lung Lesion': [
        'nodule', 'nodules', 'nodular', 'mass', 'masses', 'lesion', 'lesions',
        'cavitary', 'cavitation', 'granuloma', 'granulomas',
    ],
    'Edema': [
        'edema', 'vascular congestion', 'pulmonary congestion',
        'vascular prominence', 'fluid overload', 'chf', 'kerley',
        'vascular redistribution', 'perihilar haziness',
        'congestive heart failure', 'pulmonary venous hypertension',
    ],
    'Consolidation': [
        'consolidation', 'consolidative', 'airspace disease',
        'air space disease', 'air-space disease', 'consolidations',
    ],
    'Pneumonia': [
        'pneumonia', 'pneumonitis', 'infectious process', 'infection',
        'bronchopneumonia', 'pneumonic', 'infectious', 'infiltrative process',
        'atypical infection', 'infectious etiology', 'infectious infiltrate',
    ],
    'Atelectasis': [
        'atelectasis', 'atelectatic', 'volume loss', 'collapse',
        'subsegmental atelectasis', 'lobar collapse', 'atelectasia',
        'platelike atelectasis', 'plate-like atelectasis',
    ],
    'Pneumothorax': [
        'pneumothorax', 'pneumothoraces', 'ptx',
    ],
    'Pleural Effusion': [
        'pleural effusion', 'pleural effusions', 'effusion', 'effusions',
        'pleural fluid', 'blunting of the costophrenic',
        'costophrenic angle blunting', 'blunted costophrenic',
    ],
    'Pleural Other': [
        'pleural thickening', 'pleural scarring', 'pleural plaque',
        'pleural plaques', 'fibrosis', 'fibrotic', 'pleural calcification',
        'pleural parenchymal', 'apical capping',
    ],
    'Fracture': [
        'fracture', 'fractures', 'fractured', 'displaced rib',
    ],
    'Support Devices': [
        'endotracheal tube', 'et tube', 'ett', 'tracheostomy',
        'nasogastric tube', 'ng tube', 'orogastric', 'feeding tube',
        'chest tube', 'central line', 'central venous catheter', 'picc',
        'catheter', 'pacemaker', 'pacer', 'icd', 'defibrillator',
        'sternotomy wire', 'sternal wire', 'surgical clips', 'stent',
        'port-a-cath', 'port a cath', 'swan-ganz', 'iabp', 'lead tip',
        'line', 'lines', 'tube', 'tubes', 'wire', 'wires', 'device',
        'devices', 'hardware', 'cannula', 'drain', 'valve', 'prosthesis',
        'clips', 'leads', 'telemetry', 'sheath', 'introducer', 'pigtail',
        'cabg', 'pacing', 'catheters', 'support apparatus',
    ],
}

NEGATION = [
    'no', 'not', 'without', 'free of', 'negative for', 'absence of',
    'absent', 'resolved', 'clear of', 'no evidence of', 'no sign of',
    'rules out', 'ruled out', 'unremarkable for', 'nor',
]

UNCERTAIN = [
    'may', 'might', 'possible', 'possibly', 'probable', 'probably',
    'cannot exclude', 'cannot be excluded', 'can not be excluded',
    'suspicious for', 'questionable', 'equivocal', 'versus', ' vs ',
    'differential', 'could represent', 'suggestive of', 'concerning for',
    'likely', 'appears', 'presumed', 'if clinically',
]

# Tokens that cut a negation's scope: 'no effusion but consolidation present'
SCOPE_BREAK = [
    ' but ', ' however ', ' although ', ' though ', ' with ',
    ' there is ', ' there are ', ' demonstrat', ' shows ', ' showing ',
]

SENT_SPLIT = re.compile(r'[.;]\s*|\n')


def _normalize(text):
    if not isinstance(text, str):
        return ''
    text = text.replace('\\n', ' ').lower()
    return re.sub(r'\s+', ' ', text).strip()


def _find_all(haystack, needles):
    """Character positions of every needle occurrence, word-bounded."""
    hits = []
    for n in needles:
        pattern = r'\b' + re.escape(n).replace(r'\ ', r'\s+') + r'\b'
        hits.extend(m.start() for m in re.finditer(pattern, haystack))
    return hits


def label_text(text, label_cols=None):
    """
    Return a dict of finding -> 1.0 (present), 0.0 (absent), -1.0 (uncertain),
    or np.nan (not mentioned). Matches CheXpert's label convention.

    Negation scope runs from a negation cue to the end of the clause, so
    'no focal consolidation, pleural effusion, or pneumothorax' correctly
    negates all three.
    """
    label_cols = label_cols or LABEL_COLS
    text = _normalize(text)
    out = {c: np.nan for c in label_cols}

    if not text:
        return out

    for sent in SENT_SPLIT.split(text):
        if not sent.strip():
            continue

        neg_pos = _find_all(sent, NEGATION)
        unc_pos = _find_all(sent, UNCERTAIN)
        brk_pos = [sent.find(b) for b in SCOPE_BREAK if b in sent]
        brk_pos = [p for p in brk_pos if p >= 0]

        for finding, phrases in PHRASES.items():
            if finding not in out:
                continue
            # Longest phrases first so specific forms take precedence.
            for pos in _find_all(sent, sorted(phrases, key=len, reverse=True)):
                # A cue governs the mention if it precedes it with no
                # scope-breaking token in between.
                def governs(cues, _pos=pos):
                    return any(
                        c < _pos and not any(c < b < _pos for b in brk_pos)
                        for c in cues
                    )

                if governs(neg_pos):
                    value = 0.0
                elif governs(unc_pos) or any(pos < u < pos + 60 for u in unc_pos):
                    value = -1.0
                else:
                    value = 1.0

                prev = out[finding]
                # Positive assertions win over negatives elsewhere in the text.
                if pd.isna(prev) or value == 1.0 or (value == -1.0 and prev == 0.0):
                    out[finding] = value

    # CheXpert hierarchy: a positive child implies a positive parent.
    for child, parent in PARENT_OF.items():
        if out.get(child) == 1.0 and out.get(parent) != 1.0:
            out[parent] = 1.0

    # CheXbert sets No Finding whenever nothing else is positive, rather than
    # requiring an explicit statement of normality.
    if 'No Finding' in out:
        others = [c for c in label_cols
                  if c not in ('No Finding', 'Support Devices')]
        out['No Finding'] = 0.0 if any(out[c] == 1.0 for c in others) else 1.0

    return out


def label_frame(texts, label_cols=None):
    label_cols = label_cols or LABEL_COLS
    rows = [label_text(t, label_cols) for t in texts]
    return pd.DataFrame(rows, columns=label_cols)


# ----------------------------------------------------------------- validate

def validate(parquet, text_col, n, split):
    from sklearn.metrics import f1_score

    df = pd.read_parquet(parquet)
    if 'split' in df.columns and split != 'all':
        df = df[df['split'] == split]
    df = df.head(n).reset_index(drop=True)

    print(f"labeling {len(df)} reference impressions...\n")
    pred = label_frame(df[text_col].fillna(''))

    print("=" * 74)
    print("AGREEMENT WITH CHEXBERT ON THE SAME TEXT")
    print("=" * 74)
    header = (f"{'label':<30} {'F1':>7} {'prec':>7} {'rec':>7} "
              f"{'n_pos':>7} {'agree':>7}")
    print(header)
    print("-" * len(header))

    f1s = []
    for col in LABEL_COLS:
        truth = pd.to_numeric(df[col], errors='coerce')
        mine = pred[col]

        # Compare only where CheXbert committed to present or absent.
        mask = truth.isin([0.0, 1.0])
        if mask.sum() < 20:
            print(f"{col:<30} {'--':>7}")
            continue

        y = (truth[mask] == 1.0).astype(int)
        # Treat my uncertain and not-mentioned as absent for this comparison.
        p = (mine[mask].fillna(0.0) == 1.0).astype(int)

        if y.sum() == 0:
            print(f"{col:<30} {'no pos':>7}")
            continue

        f1 = f1_score(y, p, zero_division=0)
        tp = int(((y == 1) & (p == 1)).sum())
        prec = tp / max(int((p == 1).sum()), 1)
        rec = tp / max(int((y == 1).sum()), 1)
        agree = float((y == p).mean())
        f1s.append(f1)

        print(f"{col:<30} {f1:>7.3f} {prec:>7.3f} {rec:>7.3f} "
              f"{int(y.sum()):>7} {agree:>7.3f}")

    print(f"\nmacro F1 vs CheXbert: {np.mean(f1s):.3f}")
    print("""
How to read this. Above ~0.80 macro means this labeler reproduces CheXbert
closely enough to score generated text with, and you can quote the number as
justification. Between 0.6 and 0.8, use it for relative comparisons between
your own models but not for claims against published results. Below 0.6, the
phrase lists need work before it is worth anything.

Low recall on a label means missing surface forms; add them to PHRASES.
Low precision usually means negation scope, so check SCOPE_BREAK.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--parquet', default='data/chexpert_plus_fixed.parquet')
    ap.add_argument('--text-col', default='section_impression')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--text', default=None)
    args = ap.parse_args()

    if args.text:
        for k, v in label_text(args.text).items():
            if not pd.isna(v):
                name = {1.0: 'present', 0.0: 'absent', -1.0: 'uncertain'}[v]
                print(f"  {k:<30} {name}")
        return

    if args.validate:
        validate(args.parquet, args.text_col, args.n, args.split)
        return

    ap.print_help()


if __name__ == '__main__':
    main()
