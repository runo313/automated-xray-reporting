#!/bin/bash
python3 -m src.xray_report.evaluate \
    --checkpoint-path checkpoints/resnet_transformer/best.pt \
    --parquet-path data/chexpert_plus_clean.parquet \
    --vocab-path data/vocab.pkl \
    --image-root data/images \
    --decoder-type transformer