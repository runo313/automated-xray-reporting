#!/bin/bash
python3 -m src.xray_report.train_classifier_only \
    --parquet-path data/chexpert_plus_clean.parquet \
    --vocab-path data/vocab.pkl \
    --image-root data/images \
    --checkpoint-dir checkpoints/classifier_only \
    --batch-size 128 \
    --num-epochs 7 \
    --num-workers 4