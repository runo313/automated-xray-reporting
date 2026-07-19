# Datasets

## 1. IU X-Ray (Open-I Indiana University Chest X-Ray Collection)
- Purpose: paired image + free-text report data for the decoder (Layer 3)
- Source: NIH Open-I service
- Access: open, no credentialing required
- Status: NOT YET DOWNLOADED
- S3 location (once uploaded): s3://<bucket>/raw/iu-xray/

## 2. CheXpert (Stanford ML Group)
- Purpose: multi-label findings annotations for the findings classifier (Layer 2)
- Source: Stanford ML Group
- Access: requires free registration (email + research-use license agreement),
  self-serve, fast approval
- Status: NOT YET REGISTERED
- S3 location (once uploaded): s3://<bucket>/raw/chexpert/

## 3. NIH ChestX-ray14
- Purpose: large unlabeled/weakly-labeled image pool for self-supervised
  pretraining of the encoder
- Source: NIH Clinical Center (also mirrored on Kaggle / Box)
- Access: open, no credentialing required
- Status: NOT YET DOWNLOADED
- S3 location (once uploaded): s3://<bucket>/raw/chestxray14/

## 4. MIMIC-CXR (optional upgrade path, not committed to)
- Purpose: larger/higher-quality paired image-report alternative to IU X-Ray
- Source: PhysioNet
- Access: requires CITI human-subjects training + PhysioNet credentialing
  (can take days to weeks) - submit request early if pursuing this
- Status: NOT REQUESTED YET


