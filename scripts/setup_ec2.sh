#!/bin/bash
set -e

git clone https://github.com/runo313/automated-xray-reporting
cd automated-xray-reporting

conda env create -f environment.yml
conda activate xray-report

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"