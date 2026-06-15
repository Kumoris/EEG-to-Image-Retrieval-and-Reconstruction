#!/bin/bash
#SBATCH --job-name=msatn
#SBATCH --partition=i64m1tga40ue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

SEED=${1:-0}

source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate /hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg

echo "=== Training multiscale_blur_attn seed=${SEED} on $(hostname) ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

python -m eeg_cogcappro.train_multiscale \
    --config configs/atms_multiscale_blur_attn.yaml \
    --seed ${SEED} \
    --output-dir runs/multiscale_blur_attn_seed${SEED} \
    --device auto

echo "=== Done: seed=${SEED} ==="