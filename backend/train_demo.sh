#!/bin/bash
# Train a small demo model for MDS-ICU
# Run from backend/ directory: bash train_demo.sh

set -e
cd "$(dirname "$0")/src"

echo "=== Step 1: Generate demo dataset ==="
python generate_demo_data.py

echo ""
echo "=== Step 2: Train demo model ==="
PYTHONPATH="$(pwd)" python main_all.py --config-name config_demo

echo ""
echo "=== Done ==="
echo "Checkpoint saved in src/outputs/"
echo "Set DEMO_CHECKPOINT env var to the best_model.ckpt path before starting the API."
