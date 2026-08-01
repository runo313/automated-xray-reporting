#!/usr/bin/env python3
"""
End-to-end inference: one chest X-ray in, a structured findings summary and a
generated narrative report out.

encoder + classifier  -> checkpoints/cls_full/best.pt   (0.790 test AUC)
decoder  -> checkpoints/dec_both_v2/best.pt (best decoder run)
thresholds  -> src/xray_report/inference/thresholds.json


Usage:
    # single image, human-readable
    python3 -m src.xray_report.inference.predict --image data/images/patient64541/study1/view1_frontal.png

    # json
    python3 -m src.xray_report.inference.predict --image <path> --json

    # a folder or a text file of paths, one per line
    python3 -m src.xray_report.inference.predict --image-list images.txt


"""

import argparse
import json
import os
import sys

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.xray_report.config import DEFAULT_MAX_LEN, LABEL_COLS
from src.xray_report.models.classifier_head import ClassifierHead
from src.xray_report.models.decoders.rnn_attention import AttentionDecoder
from src.xray_report.models.decoders.transformer_decoder import TransformerDecoder
from src.xray_report.models.encoders.cnn_pretrained import PretrainedCNNEncoder
from src.xray_report.utils.vocabulary import load_vocab

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_THRESHOLDS_PATH = os.path.join(THIS_DIR, 'thresholds.json')

DEFAULT_CLASSIFIER_CKPT = 'checkpoints/cls_full/best.pt'
DEFAULT_DECODER_CKPT = 'checkpoints/dec_both_v2/best.pt'



class XRVNormalize:
    """ToTensor gives [0,1]; the pretrained encoder expects [-1024, 1024]."""

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


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"no image at {path}")
    with Image.open(path) as im:
        return eval_transform()(im)


# checkpoints

def load_classifier(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder = PretrainedCNNEncoder()
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder = encoder.to(device).eval()

    classifier = ClassifierHead(feature_dim=encoder.feature_dim, num_labels=len(LABEL_COLS))
    classifier.load_state_dict(ckpt['classifier_state_dict'])
    classifier = classifier.to(device).eval()

    return encoder, classifier, ckpt


def load_decoder(checkpoint_path, vocab, encoder_feature_dim,
                 encoder_num_regions, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get('args', {})

    token_to_idx = vocab['token_to_idx']
    pad_idx = token_to_idx.get('<pad>', 0)
    max_len = saved_args.get('max_len', DEFAULT_MAX_LEN)
    decoder_type = saved_args.get('decoder_type', 'transformer')

    if decoder_type == 'rnn':
        decoder = AttentionDecoder(
            vocab_size=len(token_to_idx),
            embed_dim=saved_args.get('embed_dim', 256),
            hidden_size=saved_args.get('hidden_size', 512),
            findings_embed_dim=saved_args.get('findings_embed_dim', 64),
            num_labels=len(LABEL_COLS),
            feature_dim=encoder_feature_dim,
        )
    else:
        decoder = TransformerDecoder(
            vocab_size=len(token_to_idx),
            embed_dim=saved_args.get('embed_dim', 256),
            num_heads=saved_args.get('num_heads', 8),
            num_layers=saved_args.get('num_layers', 4),
            num_labels=len(LABEL_COLS),
            feature_dim=encoder_feature_dim,
            max_len=max_len,
            dropout=saved_args.get('dropout', 0.1),
            pad_idx=pad_idx,
            num_regions=encoder_num_regions,
        )
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    decoder = decoder.to(device).eval()

    condition = saved_args.get('condition', 'both')
    return decoder, condition, max_len, ckpt


def load_thresholds(path):
    with open(path) as f:
        data = json.load(f)
    thresholds = data['thresholds']
    missing = [c for c in LABEL_COLS if c not in thresholds]
    if missing:
        raise ValueError(f"thresholds.json is missing labels: {missing}")
    return thresholds


# decode

def decode_tokens(ids, idx_to_token, bos, eos, pad):
    words = []
    for i in ids:
        i = int(i)
        if i == eos:
            break
        if i in (bos, pad):
            continue
        words.append(idx_to_token.get(i, '<unk>'))
    return ' '.join(words)


def condition_inputs(spatial, findings, mode):
    """Zero whichever stream the trained decoder was told to ignore. Must
    match training/train_decoder.py's condition_inputs exactly, or a decoder
    trained under e.g. 'findings' will silently get spatial features it
    never learned to use."""
    if mode == 'both':
        return spatial, findings
    if mode == 'findings':
        return torch.zeros_like(spatial), findings
    if mode == 'image':
        return spatial, torch.zeros_like(findings)
    if mode == 'none':
        return torch.zeros_like(spatial), torch.zeros_like(findings)
    raise ValueError(f"unknown condition: {mode}")


# -------------------------------------------------------------------- model

class ReportPredictor:
    """Loads everything once; call .predict(path) per image."""

    def __init__(self, classifier_checkpoint=DEFAULT_CLASSIFIER_CKPT,
                 decoder_checkpoint=DEFAULT_DECODER_CKPT,
                 vocab_path='data/vocab.pkl',
                 thresholds_path=DEFAULT_THRESHOLDS_PATH,
                 device=None, verbose=True):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.vocab = load_vocab(vocab_path)
        self.thresholds = load_thresholds(thresholds_path)

        self.encoder, self.classifier, cls_ckpt = load_classifier(
            classifier_checkpoint, self.device)
        self.decoder, self.condition, self.max_len, dec_ckpt = load_decoder(
            decoder_checkpoint, self.vocab, self.encoder.feature_dim,
            self.encoder.num_regions, self.device)

        tok = self.vocab['token_to_idx']
        self.bos, self.eos = tok['<bos>'], tok['<eos>']
        self.pad = tok.get('<pad>', 0)

        if verbose:
            print(f"[predict] device: {self.device}", file=sys.stderr)
            print(f"[predict] classifier: {classifier_checkpoint} "
                  f"(epoch {cls_ckpt.get('epoch')}, "
                  f"val AUC {cls_ckpt.get('best_val_auc')})", file=sys.stderr)
            print(f"[predict] decoder: {decoder_checkpoint} "
                  f"(condition='{self.condition}', "
                  f"epoch {dec_ckpt.get('epoch')})", file=sys.stderr)

    @torch.no_grad()
    def predict(self, image_path, max_gen_len=None):
        """
        Returns:
            dict with 'findings' (label -> {probability, positive}),
            'positive_findings' (ordered list), and 'generated_report' (str)
        """
        image = load_image(image_path).unsqueeze(0).to(self.device)

        pooled, spatial = self.encoder(image)
        cls_logits = self.classifier(pooled)
        probs = torch.sigmoid(cls_logits.float())[0]

        findings = {}
        positive = []
        for i, col in enumerate(LABEL_COLS):
            p = float(probs[i])
            is_pos = p > self.thresholds[col]
            findings[col] = {'probability': round(p, 4), 'positive': is_pos}
            if is_pos:
                positive.append(col)

        cond_findings = probs.unsqueeze(0)
        feats, cond = condition_inputs(spatial, cond_findings, self.condition)

        gen_ids, _ = self.decoder.generate(
            feats, cond, max_gen_len or self.max_len, self.bos, self.eos)
        report = decode_tokens(gen_ids[0].tolist(), self.vocab['idx_to_token'],
                               self.bos, self.eos, self.pad)

        return {
            'image': image_path,
            'findings': findings,
            'positive_findings': positive,
            'generated_report': report,
        }


# --------------------------------------------------------------------- CLI

def find_reference(image_path, parquet_path, text_col='section_impression'):
    """
    If this image is a row in the parquet (i.e. it's a known train/val/test
    image rather than a genuinely new scan), return its reference text and
    labels for comparison. Matches on the path suffix, since the parquet
    stores paths like 'train/patientX/...' while image_path may be absolute.
    """
    if not os.path.exists(parquet_path):
        return None
    df = pd.read_parquet(parquet_path)
    suffix = os.path.join(*image_path.replace('\\', '/').split('/')[-3:])
    match = df[df['path_to_image'].str.endswith(suffix)]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        'text': row.get(text_col, None),
        'labels': {c: row[c] for c in LABEL_COLS if c in row},
        'split': row.get('split', None),
    }


def print_human(result, reference=None):
    print("\n=== Chest X-Ray Report ===")
    print(f"Image: {result['image']}\n")

    print("Structured Findings (classifier):")
    positive = result['positive_findings']
    if positive:
        for col in positive:
            p = result['findings'][col]['probability']
            print(f"  {col:<28} POSITIVE  (p={p:.2f})")
    else:
        print("  (no findings above threshold)")

    negative = [c for c in LABEL_COLS if c not in positive]
    if negative:
        print(f"  [{len(negative)} more findings negative]")

    print("\nGenerated Impression (decoder):")
    print(f"  {result['generated_report']}")

    if reference:
        print("\n--- Reference (ground truth, image is in the dataset) ---")
        if reference['split']:
            print(f"split: {reference['split']}")
        if reference['text']:
            print(f"Reference impression:\n  {reference['text']}")
        ref_pos = [c for c, v in reference['labels'].items() if v == 1.0]
        print(f"Reference positive findings: {', '.join(ref_pos) or '(none)'}")

    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', help="path to a single chest X-ray image")
    ap.add_argument('--image-list',
                    help="text file of image paths, one per line")
    ap.add_argument('--classifier-checkpoint', default=DEFAULT_CLASSIFIER_CKPT)
    ap.add_argument('--decoder-checkpoint', default=DEFAULT_DECODER_CKPT)
    ap.add_argument('--vocab-path', default='data/vocab.pkl')
    ap.add_argument('--thresholds-path', default=DEFAULT_THRESHOLDS_PATH)
    ap.add_argument('--show-reference', action='store_true',
                    help="if the image is a known row in the parquet, show "
                         "its ground-truth impression and labels alongside "
                         "the prediction")
    ap.add_argument('--parquet-path', default='data/chexpert_plus_fixed.parquet',
                    help="only used with --show-reference")
    ap.add_argument('--json', action='store_true',
                    help="machine-readable output; disables --show-reference "
                         "printing extras")
    args = ap.parse_args()

    if not args.image and not args.image_list:
        ap.error("pass --image or --image-list")

    images = []
    if args.image:
        images.append(args.image)
    if args.image_list:
        with open(args.image_list) as f:
            images.extend(line.strip() for line in f if line.strip())

    predictor = ReportPredictor(
        classifier_checkpoint=args.classifier_checkpoint,
        decoder_checkpoint=args.decoder_checkpoint,
        vocab_path=args.vocab_path,
        thresholds_path=args.thresholds_path,
        verbose=not args.json,
    )

    results = []
    for path in images:
        try:
            result = predictor.predict(path)
        except FileNotFoundError as e:
            print(f"[predict] SKIP {path}: {e}", file=sys.stderr)
            continue

        reference = None
        if args.show_reference:
            reference = find_reference(path, args.parquet_path)

        if args.json:
            if reference:
                result['reference'] = reference
            results.append(result)
        else:
            print_human(result, reference)

    if args.json:
        print(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
