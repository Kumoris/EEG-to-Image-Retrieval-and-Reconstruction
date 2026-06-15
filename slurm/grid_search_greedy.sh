#!/bin/bash
#SBATCH --job-name=gridg
#SBATCH --partition=i64m1tga40ue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate /hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg

cd /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex

echo "=== Running greedy train-optimize ==="
python scripts/grid_search_ensemble.py --train-optimize --n-subsamples 10

echo "=== Done ==="