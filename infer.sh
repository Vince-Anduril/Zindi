#!/bin/bash
# Generate submission with fine-tuned model
# Usage: bash infer.sh [--no-adapter]

set -e
echo "=== Generating submission ==="

PYTHONPATH=. python scripts/infer_mlx.py \
  --model ./models/aya-expanse-8b \
  --adapter ./outputs/aya-expanse-8b-finetuned \
  --output ./outputs/submission_final.csv \
  $@

echo "=== Done! ==="
echo "Submit: outputs/submission_final.csv"
