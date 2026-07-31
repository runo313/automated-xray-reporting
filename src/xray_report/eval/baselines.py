#!/usr/bin/env python3
"""
Generation evaluation harness, plus the two baselines a decoder has to beat.

The scoring half of this file is the reusable part. score_reports() takes
generated strings and the reference label rows and returns clinical-efficacy
metrics; the decoder will call it unchanged later.

Clinical efficacy compares labels extracted from the GENERATED text against
the reference CheXbert labels already in the parquet. That avoids compounding
labeler error on both sides. BLEU is reported too, but it is capped here:
about 69% of impressions reference a prior study or interval change that no
image-only model can produce, so treat n-gram scores as context, not results.

FIXED from the first version. Scoring used to select rows with
truth.isin([0.0, 1.0]), keeping only entries where CheXbert wrote an explicit
value. On this data a blank IS the implicit negative, so that subset runs
75-98% positive (Atelectasis: 27,240 positives out of ~28,000 explicit rows).
Any model that said "yes" to everything scored near 1.0. Labels now go
through config.encode_label_matrix, the same function the training loss uses,
so blanks are negatives under BLANK_POLICY and uncertain entries follow
UNCERTAINTY_POLICY. Numbers from here are comparable to evaluate.py.

Baselines:
  constant   emit the single most common training impression for every image
  retrieval  nearest neighbour in encoder feature space, return that
             training image's impression

Usage:
    python3 -m src.xray_report.eval.baselines \
        --checkpoint checkpoints/cls_full/best.pt \
        --parquet data/chexpert_plus_fixed.parquet \
        --image-root data/images \
        --index-size 40000
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from PIL import Image
from sklearn.metrics import f1_score
from torchvision import transforms

from src.xray_report.config import LABEL_COLS, encode_label_matrix
from src.xray_report.eval.text_labeler import label_frame
from src.xray_report.models.classifier_head import ClassifierHead
from src.xray_report.models.encoders.cnn_pretrained import PretrainedCNNEncoder

SPLIT_PREFIXES = ('train/', 'valid/', 'val/', 'test/')

# The five CheXpert competition labels, for comparability with published work.
COMPETITION_FIVE = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                    'Pleural Effusion']


def strip_prefix(p):
    for prefix in SPLIT_PREFIXES:
        if p.startswith(prefix):
            return p[len(prefix):]
    return p


class XRVNormalize:
    def __call__(self, x):
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        return (2.0 * x - 1.0) * 1024.0


def eval_transform(size=224):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        XRVNormalize(),
    ])


# ===================================================================== metrics

def score_reports(generated, ref_labels, label_cols=None, references=None,
                  name=""):
    """
    Clinical efficacy of generated reports.

    Args:
        generated: list of generated report strings
        ref_labels: DataFrame of reference labels, one row per generated
                    report, in raw CheXpert convention (1.0 / 0.0 / -1.0 / NaN).
                    Row order must match `generated`; the index is ignored.
        references: optional list of reference strings, for BLEU
        name: label for the printed block

    Returns: dict of metrics
    """
    label_cols = label_cols or LABEL_COLS

    # Extracted labels from the generated text: 1.0 present, 0.0 absent,
    # -1.0 uncertain, NaN not mentioned. Only "present" counts as a positive
    # prediction, so anything the model failed to state counts as a miss.
    pred = label_frame(generated, label_cols)
    pred_arr = pred[label_cols].to_numpy(dtype=float)

    # Reference labels under the SAME convention the training loss uses.
    truth_arr, mask_arr = encode_label_matrix(
        ref_labels.reset_index(drop=True), label_cols
    )

    print("=" * 74)
    print(f"CLINICAL EFFICACY{(' - ' + name) if name else ''}")
    print("=" * 74)
    header = (f"{'label':<30} {'F1':>7} {'prec':>7} {'rec':>7} {'n_pos':>7}")
    print(header)
    print("-" * len(header))

    f1s, results = [], {}
    for i, col in enumerate(label_cols):
        keep = mask_arr[:, i] == 1.0
        if keep.sum() < 20:
            continue

        y = truth_arr[keep, i].astype(int)
        p = (np.nan_to_num(pred_arr[keep, i], nan=0.0) == 1.0).astype(int)
        if y.sum() == 0:
            continue

        f1 = f1_score(y, p, zero_division=0)
        tp = int(((y == 1) & (p == 1)).sum())
        prec = tp / max(int((p == 1).sum()), 1)
        rec = tp / max(int((y == 1).sum()), 1)
        f1s.append(f1)
        results[col] = {'f1': f1, 'precision': prec, 'recall': rec,
                        'n_pos': int(y.sum()), 'n_eval': int(keep.sum())}
        print(f"{col:<30} {f1:>7.3f} {prec:>7.3f} {rec:>7.3f} {int(y.sum()):>7}")

    macro = float(np.mean(f1s)) if f1s else float('nan')
    print(f"\nmacro clinical F1: {macro:.4f}")

    sub = [results[c]['f1'] for c in COMPETITION_FIVE if c in results]
    if sub:
        print(f"competition-five F1: {np.mean(sub):.4f}")

    out = {'macro_f1': macro, 'per_label': results}

    if references is not None:
        smooth = SmoothingFunction().method1
        scores = [
            sentence_bleu([r.lower().split()], g.lower().split(),
                          weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smooth)
            for g, r in zip(generated, references)
        ]
        out['bleu4'] = float(np.mean(scores))
        print(f"BLEU-4: {out['bleu4']:.4f}   "
              f"(capped: ~69% of references cite priors)")

    print()
    return out


# ==================================================================== features

@torch.no_grad()
def encode(paths, image_root, encoder, device, batch_size=64, log_every=20000):
    """Pooled encoder features for a list of parquet paths."""
    tf = eval_transform()
    feats, kept = [], []
    start = time.time()

    for i in range(0, len(paths), batch_size):
        imgs, ok = [], []
        for rel in paths[i:i + batch_size]:
            full = os.path.join(image_root, strip_prefix(rel))
            try:
                with Image.open(full) as im:
                    imgs.append(tf(im))
                ok.append(rel)
            except Exception:
                pass
        if not imgs:
            continue
        pooled, _ = encoder(torch.stack(imgs).to(device))
        feats.append(pooled.float().cpu())
        kept.extend(ok)

        if log_every and (i + batch_size) % log_every < batch_size:
            done = i + batch_size
            rate = done / max(time.time() - start, 1e-9)
            print(f"  {done}/{len(paths)}  {rate:.0f} img/s", flush=True)

    return torch.cat(feats), kept


def load_encoder(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = PretrainedCNNEncoder()
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder = encoder.to(device).eval()

    classifier = ClassifierHead(feature_dim=encoder.feature_dim,
                                num_labels=len(LABEL_COLS))
    classifier.load_state_dict(ckpt['classifier_state_dict'])
    classifier = classifier.to(device).eval()

    print(f"loaded {checkpoint_path} (epoch {ckpt.get('epoch')}, "
          f"val AUC {ckpt.get('best_val_auc')})")
    return encoder, classifier


# =================================================================== baselines

def retrieval_baseline(train_feats, train_texts, test_feats, chunk=256):
    """Cosine nearest neighbour; returns the matched training impression."""
    tr = torch.nn.functional.normalize(train_feats, dim=1)
    te = torch.nn.functional.normalize(test_feats, dim=1)

    out = []
    for i in range(0, len(te), chunk):
        sims = te[i:i + chunk] @ tr.T
        out.extend(train_texts[j] for j in sims.argmax(dim=1).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/cls_full/best.pt')
    ap.add_argument('--parquet', default='data/chexpert_plus_fixed.parquet')
    ap.add_argument('--image-root', default='data/images')
    ap.add_argument('--text-col', default='section_impression')
    ap.add_argument('--index-size', type=int, default=40000,
                    help="training images to index; 0 uses all")
    ap.add_argument('--test-size', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_parquet(args.parquet)

    train = df[df['split'] == 'train']
    test = df[df['split'] == 'test'].head(args.test_size).reset_index(drop=True)
    if args.index_size:
        train = train.sample(n=min(args.index_size, len(train)),
                             random_state=args.seed)
    train = train.reset_index(drop=True)

    print(f"index {len(train)} train, evaluate {len(test)} test\n")

    ref_texts = test[args.text_col].fillna('').tolist()

    # -------------------------------------------------------- constant
    modal = train[args.text_col].fillna('').value_counts().index[0]
    print(f"most common training impression ({len(modal)} chars):")
    print(f"  {modal[:120]}\n")

    score_reports([modal] * len(test), test[LABEL_COLS],
                  references=ref_texts, name="constant baseline")

    # -------------------------------------------------------- retrieval
    encoder, _ = load_encoder(args.checkpoint, device)

    print("\nencoding index...")
    tr_feats, tr_kept = encode(train['path_to_image'].tolist(),
                               args.image_root, encoder, device)
    tr_texts = (train.set_index('path_to_image')
                     .loc[tr_kept, args.text_col].fillna('').tolist())

    print("encoding test...")
    te_feats, te_kept = encode(test['path_to_image'].tolist(),
                               args.image_root, encoder, device)
    te = test.set_index('path_to_image').loc[te_kept].reset_index()

    print(f"\nretrieving {len(te_kept)} nearest neighbours...")
    retrieved = retrieval_baseline(tr_feats, tr_texts, te_feats)

    score_reports(retrieved, te[LABEL_COLS],
                  references=te[args.text_col].fillna('').tolist(),
                  name="retrieval baseline")

    # -------------------------------------------------------- clean slice
    if 'has_prior_ref' in te.columns:
        clean = ~te['has_prior_ref'].astype(bool)
        if clean.sum() >= 300:
            print(f"restricting to the {int(clean.sum())} image-derivable "
                  f"references (no prior-study citation)\n")
            score_reports([r for r, k in zip(retrieved, clean) if k],
                          te.loc[clean, LABEL_COLS],
                          references=te.loc[clean, args.text_col]
                                       .fillna('').tolist(),
                          name="retrieval, image-derivable slice")

    print("""
These are the numbers a decoder has to beat. Retrieval is the meaningful one:
it uses real image features and returns text a radiologist actually wrote, so
it is fluent and clinically plausible by construction. If a trained decoder
cannot beat it on macro clinical F1, the decoder is not adding value over
looking up a similar case.

Slice comparisons are only valid when the same labels clear the n>=20 filter
on both sides, so check n_pos before reading anything into the difference.
""")


if __name__ == '__main__':
    main()
