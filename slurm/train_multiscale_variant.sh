#!/bin/bash
#SBATCH --job-name=msatn
#SBATCH --partition=i64m1tga40ue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

CONFIG="${1:-configs/atms_multiscale_blur_attn.yaml}"
SEED="${2:-0}"
TAG="${3:-attn}"

source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate /hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg

echo "=== Training ${TAG} seed=${SEED} config=${CONFIG} ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

python -m eeg_cogcappro.train_multiscale \
    --config "$CONFIG" \
    --seed ${SEED} \
    --output-dir "runs/multiscale_${TAG}_seed${SEED}" \
    --device auto

echo "=== Done: ${TAG} seed=${SEED} ==="