#!/bin/bash
#SBATCH --job-name=eval_tta5
#SBATCH --partition=i64m1tga40ue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00

MODALITY="${1:-rn50}"
SEED="${2:-3}"

source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate /hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg

BASE="/hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex"
cd "$BASE"

if [ "$MODALITY" = "rn50" ]; then
    CACHE="cache/features_multi.pt"
    FKEY="rn50_feature"
elif [ "$MODALITY" = "vae" ]; then
    CACHE="cache/features_multi.pt"
    FKEY="vae_feature"
else
    echo "Unknown modality: $MODALITY"
    exit 1
fi

echo "=== Re-eval ${MODALITY} seed=${SEED} with TTA=5 ==="
python3 -m eeg_cogcappro.eval_atms \
    --data-dir image-eeg-data \
    --feature-cache "$CACHE" \
    --feature-key "$FKEY" \
    --ckpt "runs/deep_${MODALITY}_seed${SEED}/best.pt" \
    --split test \
    --tta-n 5 \
    --output "results/deep_${MODALITY}_seed${SEED}_test_tta5.json" \
    --device auto

echo "=== Done ${MODALITY} seed=${SEED} ==="