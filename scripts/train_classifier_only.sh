#!/usr/bin/env bash
# Train the classifier (encoder + 14-label head).
#
#   bash scripts/run_classifier.sh
#
# Flags:
#   --freeze-backbone   train the head only. Best result so far 
#   --freeze-bn         keep pretrained BatchNorm stats while fine-tuning
#   --train-subsample N cap training rows, for a quick smoke run
#   --amp               mixed precision, roughly halves epoch time
#   --num-epochs N      val macro AUC plateaus by ~5
#   --encoder-lr / --head-lr / --weight-decay / --batch-size
#
# Encoder is hardcoded in training/train_classifier.py (currently RadDino).

set -e
cd /home/runosiakpebru/automated-xray-reporting

python3 -m src.xray_report.training.train_classifier \
    --parquet-path data/chexpert_plus_fixed.parquet \
    --vocab-path data/vocab.pkl \
    --image-root data/images \
    --checkpoint-dir checkpoints/cls_raddino_frozen \
    --freeze-backbone \
    --head-lr 1e-3 \
    --num-epochs 5 \
    --amp

# Logs go to logs/classifier_only_<timestamp>.log