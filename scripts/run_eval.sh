#!/usr/bin/env bash
# Evaluate the trained models on the test set, writing plain text to results/.
#
#   bash scripts/run_eval.sh
#
# Produces:
#   results/classifier.txt   per-label AUC, F1, prevalence, macro AUC
#   results/decoder.txt      per arm: macro clinical F1, competition-five,
#                            BLEU-4, distinct count, and the shuffled-
#                            conditioning control
#
# The shuffle control generates each report from a DIFFERENT image's
# conditioning. If the score barely drops, the decoder is producing
# unconditional boilerplate rather than reading the image.
#
# Flags:
#   evaluate_classifier.py --checkpoint-path --batch-size --eval-decoder
#   eval_decoder.py        --checkpoint --test-size --no-shuffle-control

set -e
cd "$(dirname "$0")/.."
mkdir -p results

# --- classifier ------------------------------------------------------------
# This script redirects its own stdout into logs/, so copy the log it wrote.
python3 -m src.xray_report.eval.evaluate_classifier \
    --checkpoint-path checkpoints/cls_raddino_frozen/best.pt \
    --parquet-path data/chexpert_plus_fixed.parquet \
    --vocab-path data/vocab.pkl \
    --image-root data/images
cp "$(ls -t logs/eval_* | head -1)" results/classifier.txt

# --- decoder ---------------------------------------------------------------
: > results/decoder.txt
for ARM in both image findings; do
    CKPT="checkpoints/dec_raddino_${ARM}/best.pt"
    [ -f "$CKPT" ] || continue
    echo "########## condition = ${ARM} ##########" >> results/decoder.txt
    python3 -m src.xray_report.eval.eval_decoder \
        --checkpoint "$CKPT" \
        --parquet-path data/chexpert_plus_fixed.parquet \
        --image-root data/images \
        --test-size 1000 >> results/decoder.txt
done

echo "wrote results/classifier.txt results/decoder.txt"