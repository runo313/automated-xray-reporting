python3 -m src.xray_report.train \
    --parquet-path data/chexpert_plus_clean.parquet \
    --vocab-path data/vocab.pkl \
    --image-root data/images \
    --checkpoint-dir checkpoints/resnet_transformer \
    --decoder-type transformer \
    --batch-size 128 \
    --num-epochs 15 \
    --num-workers 4 \
    --log-dir logs/train_real