#!/bin/bash
set -e

mkdir -p logs

python3 -m src.xray_report.download \
    --parquet-path data/chexpert_plus_clean.parquet \
    --table-ref "aimi.chexpert_plus:5yyj:v1_0.png_train:s6cj" \
    --local-root data/images \
    --split train \
    --image-size 256 \
    --max-workers 2 \
    --log-path logs/download.log
