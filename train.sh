#!/bin/bash
# Fine-tune Aya Expanse 8B with LoRA
# Usage: bash train.sh

set -e
echo "=== Fine-tuning Aya Expanse 8B ==="

PYTHONPATH=. python -m mlx_lm.lora \
  --model ./models/aya-expanse-8b \
  --train \
  --data ./outputs/mlx_data \
  --iters 3000 \
  --batch-size 4 \
  --lora-layers 16 \
  --learning-rate 1e-5 \
  --adapter-path ./outputs/aya-expanse-8b-finetuned \
  --val-batches 25 \
  --save-every 500 \
  --grad-checkpoint

echo "=== Training complete! ==="
echo "Next: bash infer.sh"
