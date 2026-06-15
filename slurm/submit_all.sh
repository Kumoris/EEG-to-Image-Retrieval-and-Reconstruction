#!/bin/bash
# Submit pipeline jobs to HPC2 Slurm
# Usage:
#   bash slurm/submit_all.sh          # Submit all: conda setup + full pipeline
#   bash slurm/submit_all.sh setup    # Only submit conda setup job
#   bash slurm/submit_all.sh run      # Only submit pipeline (assumes conda is ready)
#   bash slurm/submit_all.sh final    # Submit final multi-encoder submission pipeline

set -euo pipefail
cd /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex
mkdir -p logs

MODE="${1:-all}"

CONDA_DIR="/hpc2hdd/home/dsaa2012_031/miniconda3"

submit_setup() {
    echo "=== Submitting Step 0: Conda environment setup ==="
    # Check if conda already exists
    if [ -d "$CONDA_DIR/envs/eeg" ]; then
        echo "Conda env 'eeg' already exists at $CONDA_DIR/envs/eeg"
        echo "Skipping setup. Use 'bash slurm/submit_all.sh run' to just run the pipeline."
        return 0
    fi
    
    # Download miniconda if not already present
    if [ ! -f /hpc2hdd/home/dsaa2012_031/miniconda.sh ]; then
        echo "Downloading Miniconda..."
        curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            -o /hpc2hdd/home/dsaa2012_031/miniconda.sh
    fi
    
    SETUP_ID=$(sbatch --parsable slurm/00_setup_conda.sh)
    echo "Setup job submitted: $SETUP_ID"
    echo "Monitor: squeue -j $SETUP_ID"
    echo "Logs: logs/00_setup_conda_${SETUP_ID}.out"
    return $SETUP_ID
}

submit_pipeline() {
    local DEPEND=""
    if [ -n "${1:-}" ]; then
        DEPEND="--dependency=afterok:$1"
    fi
    
    echo "=== Submitting full pipeline ==="
    JOB_ID=$(sbatch --parsable ${DEPEND} slurm/run_full_pipeline.sh)
    echo "Pipeline job submitted: $JOB_ID"
    echo "Monitor: squeue -j $JOB_ID"
    echo "Logs: logs/step0to4_full_${JOB_ID}.out"
    echo ""
    echo "To check progress:"
    echo "  tail -f logs/step0to4_full_${JOB_ID}.out"
    echo "  tail -f logs/step0to4_full_${JOB_ID}.err"
}

submit_final() {
    echo "=== Submitting final multi-encoder submission pipeline ==="
    JOB_ID=$(sbatch --parsable slurm/final_submission.sh)
    echo "Final job submitted: $JOB_ID"
    echo "Monitor: squeue -j $JOB_ID"
    echo "Logs:"
    echo "  tail -f logs/final_submission_${JOB_ID}.out"
    echo "  tail -f logs/final_submission_${JOB_ID}.err"
}

if [ "$MODE" = "setup" ]; then
    submit_setup
elif [ "$MODE" = "run" ]; then
    # Verify conda env exists
    if [ ! -d "$CONDA_DIR/envs/eeg" ]; then
        echo "ERROR: Conda env 'eeg' not found. Run setup first: bash slurm/submit_all.sh setup"
        exit 1
    fi
    submit_pipeline
elif [ "$MODE" = "final" ]; then
    if [ ! -d "$CONDA_DIR/envs/eeg" ]; then
        echo "ERROR: Conda env 'eeg' not found. Run setup first: bash slurm/submit_all.sh setup"
        exit 1
    fi
    submit_final
elif [ "$MODE" = "all" ]; then
    if [ -d "$CONDA_DIR/envs/eeg" ]; then
        echo "Conda already set up, submitting pipeline directly..."
        submit_pipeline
    else
        if [ ! -f /hpc2hdd/home/dsaa2012_031/miniconda.sh ]; then
            echo "Downloading Miniconda..."
            curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
                -o /hpc2hdd/home/dsaa2012_031/miniconda.sh
        fi
        SETUP_ID=$(sbatch --parsable slurm/00_setup_conda.sh)
        echo "Setup job submitted: $SETUP_ID"
        echo "Monitor: squeue -j $SETUP_ID"
        submit_pipeline "$SETUP_ID"
    fi
else
    echo "Usage: bash slurm/submit_all.sh [all|setup|run|final]"
    echo "  all:   Submit conda setup + pipeline with dependency"
    echo "  setup: Only submit conda setup job"
    echo "  run:   Only submit pipeline (requires conda env ready)"
    echo "  final: Submit final multi-encoder submission job (requires conda env ready)"
fi
