#!/bin/bash
#SBATCH --job-name=eval_ms
#SBATCH --partition=i64m1tga40ue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

CKPT="${1:-runs/multiscale_blur_seed0/best.pt}"
SPLIT="${2:-test}"
TTA="${3:-5}"
TAG="${4:-ms}"

source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate /hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg

echo "=== Evaluating ${TAG} checkpoint=${CKPT} split=${SPLIT} TTA=${TTA} ==="

python -m eeg_cogcappro.eval_multiscale \
    --checkpoint "$CKPT" \
    --split "$SPLIT" \
    --tta ${TTA} \
    --device auto

echo "=== Done: ${TAG} ==="