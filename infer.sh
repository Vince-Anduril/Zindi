#!/bin/bash
# Generate submission with fine-tuned model.
# Usage: bash infer.sh                  # uses fine-tuned adapter
#        bash infer.sh --no-adapter     # base model only
set -e

if [[ ! -d .venv ]]; then
    echo "ERROR: .venv not found. Run 'bash setup.sh' first."
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Use absolute paths — mlx-lm rejects relative paths
MODEL=${MODEL:-"$PWD/models/aya-expanse-32b-8bit"}
ADAPTER=${ADAPTER:-"$PWD/outputs/finetuned-adapter"}

if [[ ! -d "$MODEL" ]]; then
    echo "ERROR: Model directory not found: $MODEL"
    echo "Run 'bash setup.sh' first to download models."
    exit 1
fi

echo "=== Generating submission ==="

mkdir -p outputs

PYTHONPATH=. python scripts/infer_mlx.py \
    --model "$MODEL" \
    --adapter "$ADAPTER" \
    --output "$PWD/outputs/submission_final.csv" \
    "$@"

echo "=== Done ==="
echo "Submission: outputs/submission_final.csv"
