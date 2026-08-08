# Datasets

## CheXpert Plus (Stanford AIMI)

- Purpose: paired chest X-ray images, radiology report text, and multi-label
  target annotations. Used for both the clinical classifier  and
  the report decoder.
- Source: Stanford AIMI, via Redivis
- Status: used. See `src/xray_report/data/` for the download and parquet
  build scripts.

**Citation:**

[1] Stanford AIMI, "CheXpert Plus," ver. 1.0, Redivis, 2026. [Online].
    Available: https://doi.org/10.57761/fzna-pm76


## Data Location

Processed images, parquet files, vocabulary, and checkpoints are stored in
Google Cloud Storage:

```
gs://runo-cxr-data/
```

Images: https://storage.googleapis.com/runo-cxr-data/images.tar