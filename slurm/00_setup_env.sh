#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -o logs/setup_env_%j.out
#SBATCH -e logs/setup_env_%j.err
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -D /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex

echo "=== Setting up Python environment ==="
echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"

mkdir -p logs

# Install packages to user site
export PYTHONNOUSERSITE=0
pip_target="$HOME/.local"
mkdir -p "$pip_target"

echo "--- Installing PyTorch with CUDA ---"
python3 -m pip install --user --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -5

echo "--- Installing project dependencies ---"
python3 -m pip install --user --no-cache-dir \
    numpy Pillow PyYAML tqdm pytest open_clip_torch \
    timm diffusers transformers accelerate 2>&1 | tail -5

echo "--- Verifying installations ---"
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
python3 -c "import open_clip; print('open_clip ok')"
python3 -c "import timm; print('timm ok')"
python3 -c "import diffusers; print('diffusers ok')"
python3 -c "import transformers; print('transformers ok')"

echo "=== Environment setup complete at $(date) ==="