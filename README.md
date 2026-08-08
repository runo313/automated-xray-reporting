# Automated Chest X-Ray Reporting

A target-conditioned report generation system for chest radiographs. 
Given a chest X-ray, the system predicts the presence of key clinical conditions 
and generates a natural-language report describing them.

## Pipeline

1. **Encoder** — extracts image features. Two variants were built and
   compared:
   - **DenseNet-121**, fine-tuned on chest X-rays.
   - **RAD-DINO**, a self-supervised Vision Transformer pretrained on
     unlabeled chest X-rays (loaded from Hugging Face), used with its
     weights frozen after pretraining.
2. **Clinical classifier** — multi-label head on the encoder's pooled
   features, predicts presence/absence/uncertainty across 14 clinical
   findings.
3. **Report decoder** — generates report text conditioned on the classifier's
   predicted findings and the encoder's spatial feature grid. Two decoder
   architectures were built and compared:
   - A GRU-based recurrent decoder with attention over spatial features.
   - A Transformer decoder.

Training is staged: the encoder and classifier are trained and validated
first, then frozen before the decoder is trained on top. Encoder and decoder
combinations are trained and evaluated separately using the scripts in

## Repository Layout

- `src/xray_report/data/` — dataset download, parquet rebuild, vocabulary
- `src/xray_report/training/` — classifier, decoder, and joint training
- `src/xray_report/eval/` — classifier evaluation, generation baselines,
  ablation and shuffle-conditioning tests
- `src/xray_report/diagnostics/` — data/label sanity checks
- `src/xray_report/inference/` — single-image prediction script
- `scripts/` — entry-point shell scripts (see below)
- `results/` — evaluation output as plain text

## Running

```bash
bash scripts/run_classifier.sh   # train the classifier
bash scripts/run_decoder.sh      # train the decoder (arg: condition, default "both")
bash scripts/run_eval.sh         # evaluate classifier + decoder, writes results/*.txt
bash scripts/run_predict.sh      # generate sample reports, writes results/samples.txt
```
