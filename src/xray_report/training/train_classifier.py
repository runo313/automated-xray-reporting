"""
Trains Layer 1 (encoder). Reads a config YAML (see configs/encoder_finetune.yaml).
TODO: implement once data pipeline (Dataset classes) exists and is verified.
"""

import argparse
import yaml


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
