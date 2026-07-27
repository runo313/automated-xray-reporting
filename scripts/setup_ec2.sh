#!/bin/bash
set -e

cd automated-xray-reporting

python3 -m venv xray-env
source xray-env/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "Environment ready"