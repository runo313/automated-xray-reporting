# Automated Chest X-Ray Reporting

A findings-conditioned report generation system for chest radiographs. Given
a chest X-ray, the system predicts the presence of clinical findings and
generates a short natural-language report describing them.

## Pipeline

1. **Encoder** — CNN backbone, ImageNet-pretrained, fine-tuned on chest
   X-rays. Self-supervised pretraining on unlabeled X-rays planned as an
   extension.
2. **Findings classifier** — multi-label head on encoder features, predicts
   presence/absence of a fixed set of clinical findings.
3. **Report decoder** — attention-based decoder over the encoder's spatial
   feature grid, conditioned on findings classifier output, generates report
   text.
4. **API** — inference service returning findings, confidence scores, and
   generated report text for a given image.

