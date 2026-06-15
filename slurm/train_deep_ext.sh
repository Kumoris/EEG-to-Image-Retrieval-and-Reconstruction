#!/bin/bash
#SBATCH --job-name=deep_ext
#SBATCH --partition=i64m1tga40ue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/deep_ext_%j.out
#SBATCH --error=logs/deep_ext_%j.err

set -euo pipefail

BASE="/hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex"
cd "$BASE"
source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate /hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg

MODALITY="${1:-rn50}"
SEED="${2:-3}"

echo "=== Training deep_${MODALITY} seed=${SEED} on $(hostname) ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

mkdir -p results

if [ "$MODALITY" = "rn50" ]; then
    CONFIG="configs/atms_rn50.yaml"
    CACHE="cache/features_multi.pt"
    FKEY="rn50_feature"
    EPOCHS=50
elif [ "$MODALITY" = "vae" ]; then
    CONFIG="configs/atms_vae.yaml"
    CACHE="cache/features_multi.pt"
    FKEY="vae_feature"
    EPOCHS=50
else
    echo "Unknown modality: $MODALITY"
    exit 1
fi

RUNDIR="runs/deep_${MODALITY}_seed${SEED}"
RESULT_LOGITS="results/deep_${MODALITY}_seed${SEED}_test_tta5.logits.pt"

if [ -f "$RESULT_LOGITS" ]; then
    echo "SKIP seed=$SEED ($RESULT_LOGITS exists)"
    exit 0
fi

echo "=== Train ${MODALITY} seed=${SEED} === $(date)"
python3 -m eeg_cogcappro.train_atms \
    --config "$CONFIG" \
    --data-dir image-eeg-data \
    --feature-cache "$CACHE" \
    --feature-key "$FKEY" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --output-dir "$RUNDIR" \
    --device auto

echo "=== Eval ${MODALITY} seed=${SEED} === $(date)"
python3 -m eeg_cogcappro.eval_atms \
    --data-dir image-eeg-data \
    --feature-cache "$CACHE" \
    --feature-key "$FKEY" \
    --ckpt "$RUNDIR/best.pt" \
    --split test \
    --tta-n 5 \
    --output "results/deep_${MODALITY}_seed${SEED}_test.json" \
    --device auto

echo "=== Done: ${MODALITY} seed=${SEED} === $(date)"