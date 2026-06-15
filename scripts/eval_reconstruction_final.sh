#!/usr/bin/env bash
set -euo pipefail

python -m eeg_cogcappro.eval_reconstruction \
  --fake-dir recons/atms_multimodal_final \
  --output results/atms_multimodal_final_reconstruction_eval.json
