#!/bin/bash
#SBATCH --job-name=mscale
#SBATCH --partition=i64m1tga40ue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/multiscale_seed%a.out
#SBATCH --error=logs/multiscale_seed%a.err

SEED=${1:-0}

source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate /hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg

echo "=== Training multiscale_blur seed=${SEED} on $(hostname) ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -m eeg_cogcappro.train_multiscale \
    --config configs/atms_multiscale_blur.yaml \
    --seed ${SEED} \
    --output-dir runs/multiscale_blur_seed${SEED} \
    --device auto

echo "=== Done: seed=${SEED} ==="