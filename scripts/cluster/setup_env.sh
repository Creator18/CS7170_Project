#!/bin/bash
# Build the CS 7170 Python env and fetch model weights on Explorer scratch.
# Home is full (75 GB quota), so every cache points at scratch. Scratch is purged monthly:
# rerun this script to rebuild. Run as a CPU job:
#   sbatch -J cs7170-env -p short -c 4 --mem=16G -t 03:00:00 -o ~/cs7170/logs/env_%j.out ~/cs7170/tools/setup_env.sh
set -euo pipefail
S=/scratch/basu.ang/cs7170
export CONDA_PKGS_DIRS=$S/cache/conda_pkgs PIP_CACHE_DIR=$S/cache/pip HF_HOME=$S/hf XDG_CACHE_HOME=$S/cache
mkdir -p $S/envs $S/cache $S/hf
module load miniconda3/25.9.1
if [ ! -x $S/envs/cs7170/bin/python ]; then
  # conda-forge only: Anaconda's default channels need a Terms of Service acceptance
  conda create -y -p $S/envs/cs7170 --override-channels -c conda-forge python=3.11
fi
PY=$S/envs/cs7170/bin/python
$PY -m pip install --upgrade pip
# cu126 wheels still include Volta (sm_70) kernels for the V100s
$PY -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
$PY -m pip install "transformers==4.57.1" "accelerate==1.10.1" "huggingface_hub>=0.34,<1.0" pillow av "qwen-vl-utils==0.0.14" numpy
$PY - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda build", torch.version.cuda, "archs", torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else "?")
print("transformers", transformers.__version__)
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-VL-4B-Instruct")
print("weights at", p)
EOF
$PY -m pip freeze > $S/envs/cs7170.freeze.txt
echo ENV_OK
