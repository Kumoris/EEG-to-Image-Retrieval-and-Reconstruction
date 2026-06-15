#!/bin/bash
#SBATCH --job-name=eval_ms
#SBATCH --partition=i64m1tga40ue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

SEED=${1:-0}
TTA=${2:-5}

source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate /hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg

echo "=== Evaluating multiscale_blur seed=${SEED} TTA=${TTA} ==="

python -m eeg_cogcappro.eval_multiscale \
    --checkpoint runs/multiscale_blur_seed${SEED}/best.pt \
    --split test \
    --tta ${TTA} \
    --device auto

echo "=== Done: eval seed=${SEED} ==="