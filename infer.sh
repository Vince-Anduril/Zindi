#!/bin/bash
# Generate submission with fine-tuned model.
# Usage: bash infer.sh            # uses fine-tuned adapter
#        bash infer.sh --no-adapter  # uses base model only
set -e

if [[ ! -d .venv ]]; then
    echo "ERROR: .venv not found. Run 'bash setup.sh' first."
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

MODEL=${MODEL:-./models/aya-expanse-32b-4bit}
ADAPTER=${ADAPTER:-./outputs/finetuned-adapter}

echo "=== Generating submission ==="

PYTHONPATH=. python scripts/infer_mlx.py \
    --model "$MODEL" \
    --adapter "$ADAPTER" \
    --output ./outputs/submission_final.csv \
    "$@"

echo "=== Done ==="
echo "Submission: outputs/submission_final.csv"
