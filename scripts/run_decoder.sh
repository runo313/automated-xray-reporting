#!/usr/bin/env bash
# Train the report decoder on top of a trained classifier.
#
#   bash scripts/run_decoder.sh            # condition=both
#   bash scripts/run_decoder.sh image      # ablation arm
#   bash scripts/run_decoder.sh findings
#   bash scripts/run_decoder.sh none       # unconditional floor
#
# Flags:
#   --condition          both | image | findings | none.
#   --decoder-type       transformer | rnn
#   --findings-source    gt | pred | scheduled.
#   --unfreeze-encoder   also fine-tune the encoder 
#   --gen-eval-size N    val images to free-run generate for clinical F1
#   --num-layers / --num-heads / --embed-dim / --dropout
#   --amp, --train-subsample, --num-epochs, --batch-size

set -e
cd "$(dirname "$0")/.."

CONDITION="${1:-both}"

python3 -m src.xray_report.training.train_decoder \
    --parquet-path data/chexpert_plus_fixed.parquet \
    --vocab-path data/vocab.pkl \
    --image-root data/images \
    --classifier-checkpoint checkpoints/cls_raddino_frozen/best.pt \
    --checkpoint-dir "checkpoints/dec_raddino_${CONDITION}" \
    --condition "$CONDITION" \
    --decoder-type transformer \
    --num-epochs 8 \
    --amp

# Logs go to logs/dec_<type>_<condition>_<timestamp>.log