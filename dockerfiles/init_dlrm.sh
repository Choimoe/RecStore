#!/bin/bash
cd "$(dirname "$0")"
set -x
set -e

# User and project settings
USER="$(whoami)"
PROJECT_PATH="$(cd .. && pwd)"
DLRM_PATH="${PROJECT_PATH}/model_zoo/torchrec_dlrm"
TORCHREC_PATH="${PROJECT_PATH}/third_party/torchrec"

# Install system dependencies
sudo apt update
sudo apt install -y python3.10-venv git

# Clean old environment
cd ${DLRM_PATH}
# rm -rf dlrm_venv

# Create new virtual environment
python3 -m venv dlrm_venv
source ${DLRM_PATH}/dlrm_venv/bin/activate
pip install --upgrade pip

# Install PyTorch nightly for CUDA 11.8
# https://docs.pytorch.org/torchrec/setup-torchrec.html
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install fbgemm-gpu==1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torchmetrics==1.0.3
pip install torchrec==1.0 --index-url https://download.pytorch.org/whl/cu118


# Usage
# torchrun --nnodes 1 --nproc_per_node 2 --rdzv_backend c10d --rdzv_endpoint localhost --rdzv_id 54321 --role trainer dlrm_main.py