#!/bin/bash
#SBATCH -p debug
#SBATCH -o logs/00_setup_conda_%j.out
#SBATCH -e logs/00_setup_conda_%j.err
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH -D /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex

CONDA_DIR="/hpc2hdd/home/dsaa2012_031/miniconda3"
ENV_NAME="eeg"
ENV_DIR="${CONDA_DIR}/envs/${ENV_NAME}"

echo "=== Setting up Python environment ==="
echo "Started at $(date)"
echo "Node: $(hostname)"

# Step 1: Accept conda TOS
echo "--- Accepting conda TOS ---"
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>&1 || true

# Step 2: Create conda environment with Python 3.10
echo "--- Creating conda env: $ENV_NAME ---"
if [ -d "$ENV_DIR" ]; then
    echo "Environment already exists: $ENV_DIR"
else
    conda create -y -n "$ENV_NAME" python=3.10 2>&1
fi

# Step 3: Activate and install packages
echo "--- Activating environment ---"
conda activate "$ENV_NAME"

echo "--- Installing PyTorch with CUDA 11.8 ---"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -10

echo "--- Installing project dependencies ---"
pip install numpy Pillow PyYAML tqdm pytest open_clip_torch timm diffusers transformers accelerate 2>&1 | tail -10

echo "--- Verifying installation ---"
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
python3 -c "import open_clip; print('open_clip ok')"
python3 -c "import timm; print('timm ok')"
python3 -c "import diffusers; print('diffusers ok')"
python3 -c "import transformers; print('transformers ok')"
python3 -c "import yaml; print('yaml ok')"

echo "=== Setup complete at $(date) ==="
echo "Conda env path: $ENV_DIR"
echo "To activate: source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate $ENV_NAME"