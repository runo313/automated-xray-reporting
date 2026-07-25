#!/usr/bin/env python3
import sys
LABEL_COLS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices', 'No Finding',
]

DEFAULT_MAX_LEN = 50
DEFAULT_MIN_FREQ = 5
DEFAULT_IMAGE_SIZE = 224

def redirect_output(log_path):
    """Redirect stdout and stderr to the same log file."""
    log_file = open(log_path, 'w', buffering=1)   # line-buffered, so tail -f updates promptly
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file