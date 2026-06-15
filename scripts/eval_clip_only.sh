#!/bin/bash
#SBATCH -p i64m1tga800ue
#SBATCH -o logs/eval_clip_%j.out
#SBATCH -e logs/eval_clip_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -D /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex

export PATH="/hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg/bin:$PATH"

echo "=== Evaluating CLIP for diffusion_img2img_train_source ==="
python -m eeg_cogcappro.eval_reconstruction_official \
    --fake-dir recons/experiments/diffusion_img2img_train_source \
    --data-dir image-eeg-data \
    --output results/reconstruction_experiments/diffusion_img2img_train_source.json \
    --metrics requested \
    --batch-size 32 \
    --device auto \
    --allow-open-clip-fallback

echo "=== Evaluating CLIP for diffusion_prompt ==="
python -m eeg_cogcappro.eval_reconstruction_official \
    --fake-dir recons/experiments/diffusion_prompt \
    --data-dir image-eeg-data \
    --output results/reconstruction_experiments/diffusion_prompt.json \
    --metrics requested \
    --batch-size 32 \
    --device auto \
    --allow-open-clip-fallback

echo "=== Evaluating CLIP for concept_train_nearest (re-verify) ==="
# Only if concept_train_nearest directory exists in recons/experiments
if [ -d "recons/experiments/concept_train_nearest" ]; then
    python -m eeg_cogcappro.eval_reconstruction_official \
        --fake-dir recons/experiments/concept_train_nearest \
        --data-dir image-eeg-data \
        --output results/reconstruction_experiments/concept_train_nearest.json \
        --metrics requested \
        --batch-size 32 \
        --device auto \
        --allow-open-clip-fallback
else
    echo "concept_train_nearest directory not found in recons/experiments, skipping"
fi

echo "=== Re-running select_best_reconstruction ==="
python scripts/select_best_reconstruction.py \
    --experiments-root recons/experiments \
    --results-root results/reconstruction_experiments \
    --output-dir recons/atms_multimodal_final_improved \
    --summary results/reconstruction_experiments_summary.json

echo "=== All done ==="
