#!/usr/bin/env python3
"""
Evaluate trained decoders on the test set, and measure whether the
conditioning is actually being used.

The four ablation arms came out within 0.02 of each other on validation
(image 0.271, both 0.283, findings 0.294) while the deterministic template
scores 0.414 from the same classifier. If the image stream and the findings
stream each carried independent signal, 'both' should beat either alone. It
does not, which suggests the decoder is generating fluent boilerplate and
largely ignoring what it is conditioned on.

The shuffle control settles that. Generate each report from a DIFFERENT
image's conditioning, holding everything else fixed:

    real    condition on image i, score against reference i
    shuffle condition on image perm(i), score against reference i

If shuffle scores nearly as well as real, the conditioning contributes almost
nothing and the arm is a language model with decoration. A large drop means
the conditioning is doing real work and the ceiling lies elsewhere.

Usage:
    python3 -m src.xray_report.eval.eval_decoder \
        --checkpoint checkpoints/dec_both/best.pt --test-size 1000

    # every arm at once
    for C in none image findings both; do
      python3 -m src.xray_report.eval.eval_decoder \
          --checkpoint checkpoints/dec_$C/best.pt --test-size 1000
    done
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.xray_report.config import DEFAULT_MAX_LEN, LABEL_COLS
from src.xray_report.dataloader import build_dataloaders
from src.xray_report.eval.baselines import score_reports
from src.xray_report.models.classifier_head import ClassifierHead
from src.xray_report.models.decoders.rnn_attention import AttentionDecoder
from src.xray_report.models.decoders.transformer_decoder import TransformerDecoder
from src.xray_report.models.encoders.cnn_pretrained import PretrainedCNNEncoder
from src.xray_report.models.encoders.vit_pretrained import RadDinoEncoder
from src.xray_report.utils.vocabulary import load_vocab


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
    if mode == 'both':
        return spatial, findings
    if mode == 'findings':
        return torch.zeros_like(spatial), findings
    if mode == 'image':
        return spatial, torch.zeros_like(findings)
    if mode == 'none':
        return torch.zeros_like(spatial), torch.zeros_like(findings)
    raise ValueError(mode)


def build_from_checkpoint(path, vocab, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved = ckpt.get('args', {})

    #encoder = PretrainedCNNEncoder()
    encoder = RadDinoEncoder(pool_to=7)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    classifier = ClassifierHead(feature_dim=encoder.feature_dim,
                               num_labels=len(LABEL_COLS))
    classifier.load_state_dict(ckpt['classifier_state_dict'])

    token_to_idx = vocab['token_to_idx']
    pad_idx = token_to_idx.get('<pad>', 0)
    max_len = saved.get('max_len', DEFAULT_MAX_LEN)

    if saved.get('decoder_type', 'transformer') == 'rnn':
        decoder = AttentionDecoder(
            vocab_size=len(token_to_idx),
            embed_dim=saved.get('embed_dim', 256),
            hidden_size=saved.get('hidden_size', 512),
            findings_embed_dim=saved.get('findings_embed_dim', 64),
            num_labels=len(LABEL_COLS),
            feature_dim=encoder.feature_dim,
        )
    else:
        decoder = TransformerDecoder(
            vocab_size=len(token_to_idx),
            embed_dim=saved.get('embed_dim', 256),
            num_heads=saved.get('num_heads', 8),
            num_layers=saved.get('num_layers', 4),
            num_labels=len(LABEL_COLS),
            feature_dim=encoder.feature_dim,
            max_len=max_len,
            dropout=saved.get('dropout', 0.1),
            pad_idx=pad_idx,
            num_regions=encoder.num_regions,
        )
    decoder.load_state_dict(ckpt['decoder_state_dict'])

    print(f"loaded {path}")
    print(f"  epoch {ckpt.get('epoch')}  condition='{saved.get('condition')}'"
          f"  decoder={saved.get('decoder_type')}"
          f"  best val metric {ckpt.get('best_metric')}")

    return (encoder.to(device).eval(), classifier.to(device).eval(),
            decoder.to(device).eval(), saved, max_len)


@torch.no_grad()
def generate_all(loader, encoder, classifier, decoder, vocab, mode, max_len,
                 device, limit, shuffle_conditioning=False, seed=0):
    token_to_idx = vocab['token_to_idx']
    idx_to_token = vocab['idx_to_token']
    bos, eos = token_to_idx['<bos>'], token_to_idx['<eos>']
    pad = token_to_idx.get('<pad>', 0)

    g = torch.Generator(device='cpu').manual_seed(seed)
    texts, seen = [], 0

    for images, labels, mask, text_ids in loader:
        if seen >= limit:
            break
        images = images.to(device, non_blocking=True)

        pooled, spatial = encoder(images)
        cls_logits = classifier(pooled)
        findings = torch.sigmoid(cls_logits.float())

        if shuffle_conditioning and images.size(0) > 1:
            # Derangement-ish: roll by a random non-zero offset so no sample
            # keeps its own conditioning.
            offset = int(torch.randint(1, images.size(0), (1,),
                                       generator=g).item())
            spatial = torch.roll(spatial, shifts=offset, dims=0)
            findings = torch.roll(findings, shifts=offset, dims=0)

        feats, cond = condition_inputs(spatial, findings, mode)
        gen, _ = decoder.generate(feats, cond, max_len, bos, eos)

        for row in gen:
            texts.append(decode_tokens(row.tolist(), idx_to_token,
                                       bos, eos, pad))
        seen += images.size(0)

    return texts[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--parquet-path', default='data/chexpert_plus_fixed.parquet')
    ap.add_argument('--vocab-path', default='data/vocab.pkl')
    ap.add_argument('--image-root', default='data/images')
    ap.add_argument('--text-col', default='section_impression')
    ap.add_argument('--test-size', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--no-shuffle-control', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = load_vocab(args.vocab_path)
    df = pd.read_parquet(args.parquet_path)

    encoder, classifier, decoder, saved, max_len = build_from_checkpoint(
        args.checkpoint, vocab, device)
    mode = saved.get('condition', 'both')

    loaders = build_dataloaders(
        df, args.image_root, vocab['token_to_idx'],
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_len=max_len,
    )
    test_loader = loaders['test']
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    # -------------------------------------------------------------- real
    print(f"\ngenerating {args.test_size} test reports (real conditioning)...")
    real = generate_all(test_loader, encoder, classifier, decoder, vocab,
                        mode, max_len, device, args.test_size)
    sub = test_df.iloc[:len(real)]

    print(f"\ndistinct: {len(set(real))} of {len(real)}   "
          f"mean length {np.mean([len(t.split()) for t in real]):.1f} words")
    print("samples:")
    for t in real[:3]:
        print(f"  {t[:160]}")
    print()

    res_real = score_reports(real, sub[LABEL_COLS],
                             references=sub[args.text_col].fillna('').tolist(),
                             name=f"{mode} — real conditioning")

    if args.no_shuffle_control or mode == 'none':
        return

    # ----------------------------------------------------------- shuffled
    print(f"generating {args.test_size} test reports (SHUFFLED "
          f"conditioning)...")
    shuf = generate_all(test_loader, encoder, classifier, decoder, vocab,
                        mode, max_len, device, args.test_size,
                        shuffle_conditioning=True)
    shuf = shuf[:len(real)]

    res_shuf = score_reports(shuf, sub[LABEL_COLS],
                             references=sub[args.text_col].fillna('').tolist(),
                             name=f"{mode} — shuffled conditioning")

    # ------------------------------------------------------------ verdict
    drop = res_real['macro_f1'] - res_shuf['macro_f1']
    rel = drop / max(res_real['macro_f1'], 1e-9)

    print("=" * 74)
    print("IS THE CONDITIONING BEING USED?")
    print("=" * 74)
    print(f"  real      macro F1 {res_real['macro_f1']:.4f}")
    print(f"  shuffled  macro F1 {res_shuf['macro_f1']:.4f}")
    print(f"  drop      {drop:.4f}  ({rel:.1%} of the real score)")

    if rel < 0.15:
        print("""
  The decoder scores nearly as well on someone else's conditioning as on
  its own. It is a language model with decoration: almost all of its score
  comes from emitting statistically likely radiology text, not from reading
  this image. Fix the conditioning before tuning anything else.""")
    elif rel < 0.40:
        print("""
  Partial. The conditioning contributes, but a large share of the score is
  unconditional boilerplate. Strengthening the conditioning path is likely
  to pay off more than more epochs.""")
    else:
        print("""
  The conditioning is doing real work. The gap to the template baseline is
  then about generation capacity or training budget, not about the decoder
  ignoring its inputs.""")
    print()


if __name__ == '__main__':
    main()
