#!/bin/bash
#SBATCH -p i64m1tga800ue
#SBATCH -o logs/diffusion_img2img_%j.out
#SBATCH -e logs/diffusion_img2img_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -D /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex

echo "=== Diffusion img2img reconstruction ==="
echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python3 2>/dev/null || which python)"

# Use conda environment
export PATH="/hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg/bin:$PATH"
export PYTHONNOUSERSITE=0

# Verify GPU
nvidia-smi || { echo "ERROR: No GPU found!"; exit 1; }

# Verify packages
python -c "import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')" || { echo "ERROR: torch not available"; exit 1; }
python -c "import diffusers; print(f'diffusers {diffusers.__version__}')" || { echo "ERROR: diffusers not available"; exit 1; }

mkdir -p logs recons/experiments

# Set HuggingFace cache to a writable location
export HF_HOME="/hpc2hdd/home/dsaa2012_031/.cache/huggingface"
export HF_ENDPOINT="https://huggingface.co"

echo "HF_HOME=$HF_HOME"

echo "=== Step 1: Generate diffusion_img2img_train_source reconstructions ==="

# Use larger steps for better quality, and a moderate strength
python -m eeg_cogcappro.reconstruct_experiments \
    --method diffusion_img2img_train_source \
    --data-dir image-eeg-data \
    --feature-cache cache/features_vitl.pt \
    --retrieval-logits results/ensemble_eval_opt9mod/retrieval_test_logits.pt \
    --output-dir recons/experiments/diffusion_img2img_train_source \
    --feature-key image_clean_feature \
    --topk 5 \
    --image-size 256 \
    --diffusion-model stabilityai/sdxl-turbo \
    --diffusion-steps 4 \
    --strength 0.55 \
    --seed 20260427 \
    --device auto

IMG2IMG_EXIT=$?
echo "=== Step 1 exit code: $IMG2IMG_EXIT ==="

echo "=== Step 2: Generate diffusion_prompt reconstructions ==="

python -m eeg_cogcappro.reconstruct_experiments \
    --method diffusion_prompt \
    --data-dir image-eeg-data \
    --feature-cache cache/features_vitl.pt \
    --retrieval-logits results/ensemble_eval_opt9mod/retrieval_test_logits.pt \
    --output-dir recons/experiments/diffusion_prompt \
    --feature-key image_clean_feature \
    --topk 5 \
    --image-size 256 \
    --diffusion-model stabilityai/sdxl-turbo \
    --diffusion-steps 4 \
    --guidance-scale 0.0 \
    --seed 20260427 \
    --device auto

PROMPT_EXIT=$?
echo "=== Step 2 exit code: $PROMPT_EXIT ==="

echo "=== Step 3: Evaluate diffusion reconstructions ==="

if [ -d "recons/experiments/diffusion_img2img_train_source" ]; then
    python -m eeg_cogcappro.eval_reconstruction_official \
        --reconstruction-dir recons/experiments/diffusion_img2img_train_source \
        --data-dir image-eeg-data \
        --output results/reconstruction_experiments/diffusion_img2img_train_source.json \
        --device auto 2>/dev/null || echo "WARNING: img2img eval had issues"
else
    echo "WARNING: img2img output not found, skipping eval"
fi

if [ -d "recons/experiments/diffusion_prompt" ]; then
    python -m eeg_cogcappro.eval_reconstruction_official \
        --reconstruction-dir recons/experiments/diffusion_prompt \
        --data-dir image-eeg-data \
        --output results/reconstruction_experiments/diffusion_prompt.json \
        --device auto 2>/dev/null || echo "WARNING: prompt eval had issues"
else
    echo "WARNING: prompt output not found, skipping eval"
fi

echo "=== Step 4: Select best reconstruction ==="
python scripts/select_best_reconstruction.py \
    --experiments-root recons/experiments \
    --results-root results/reconstruction_experiments \
    --output-dir recons/atms_multimodal_final_improved \
    --summary results/reconstruction_experiments_summary.json

echo "=== Step 5: Package final submission ==="
bash scripts/package_final_submission.sh 2>/dev/null || echo "Packaging may need manual step"

echo "=== All done at $(date) ==="