#!/bin/bash
python3 -m src.xray_report.train \
    --parquet-path data/chexpert_plus_clean.parquet \
    --vocab-path data/vocab.pkl \
    --image-root data/images \
    --checkpoint-dir checkpoints/resnet_rnn_baseline_v2 \
    --lambda-weight 0.1 \
    --batch-size 128 \
    --num-epochs 15 \
    --num-workers 4 \
    --log-dir logs/train_real