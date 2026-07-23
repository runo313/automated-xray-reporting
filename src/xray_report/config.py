#!/usr/bin/env python3

LABEL_COLS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices', 'No Finding',
]

DEFAULT_MAX_LEN = 50
DEFAULT_MIN_FREQ = 5
DEFAULT_IMAGE_SIZE = 224