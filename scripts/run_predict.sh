#!/usr/bin/env bash
# Generate example reports end to end: image -> findings + narrative.
#
#   bash scripts/run_predict.sh
#
# Produces results/samples.txt with three test-set reports, each showing the
# classifier's structured findings, the decoder's generated impression, and
# the ground-truth impression for comparison.
#
# Flags:
#   --image / --image-list   single path, or a file of paths one per line
#   --show-reference include ground truth when the image is a known parquet row
#   --json     machine-readable instead of the printed layout
#   --classifier-checkpoint / --decoder-checkpoint / --thresholds-path
#
# predict.py must build the same encoder its checkpoints were trained
# with. Swap PretrainedCNNEncoder for RadDinoEncoder(pool_to=7) in
# inference/predict.py before running this against RAD-DINO checkpoints.

set -e
cd /home/runosiakpebru/automated-xray-reporting

# Three test-set images. Replace with any paths under data/images.
cat > /tmp/sample_images.txt <<'EOF'
data/images/patient19460/study1/view1_frontal.png
data/images/patient17829/study5/view1_frontal.png
data/images/patient18068/study3/view1_frontal.png
EOF

python3 -m src.xray_report.inference.predict \
    --image-list /tmp/sample_images.txt \
    --classifier-checkpoint checkpoints/cls_raddino_frozen/best.pt \
    --decoder-checkpoint checkpoints/dec_raddino_both/best.pt \
    --parquet-path data/chexpert_plus_fixed.parquet \
    --show-reference > results/samples1.txt

echo "wrote results/samples.txt"
