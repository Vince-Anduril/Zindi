#!/bin/bash
# Fine-tune with LoRA. Defaults to Aya Expanse 32B 4-bit.
# Override with: MODEL=./models/Meta-Llama-3.3-70B-Instruct-4bit bash train.sh
set -e

if [[ ! -d .venv ]]; then
    echo "ERROR: .venv not found. Run 'bash setup.sh' first."
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

MODEL=${MODEL:-./models/aya-expanse-32b-4bit}
ADAPTER=${ADAPTER:-./outputs/finetuned-adapter}

echo "=== Fine-tuning ==="
echo "Model:   $MODEL"
echo "Adapter: $ADAPTER"

mkdir -p outputs

PYTHONPATH=. python -m mlx_lm.lora \
    --model "$MODEL" \
    --train \
    --data ./outputs/mlx_data \
    --iters 3000 \
    --batch-size 4 \
    --num-layers 16 \
    --learning-rate 1e-5 \
    --adapter-path "$ADAPTER" \
    --val-batches 25 \
    --save-every 500 \
    --grad-checkpoint \
    --seed 42

echo "=== Training complete ==="
echo "Adapter saved to: $ADAPTER"
echo "Next: bash infer.sh"
