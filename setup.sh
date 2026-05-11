#!/bin/bash
# Run once on Mac Studio after SSH connection
# Usage: bash setup.sh
set -e

echo "=== Setup Zindi NLP Challenge — Mac Studio ==="

# 1. Python dependencies
echo "[1/4] Installing Python packages..."
pip install --upgrade pip -q
pip install mlx-lm huggingface_hub rouge-score pandas tqdm numpy -q

# 2. Verify MLX install + check Apple Silicon
echo "[2/4] Verifying MLX installation..."
python -c "import mlx.core as mx; print(f'MLX version OK. Default device: {mx.default_device()}')"

# 3. Download pre-quantized 4-bit MLX models (fast download + fast inference)
echo "[3/4] Downloading models (this takes 20-40 min depending on bandwidth)..."
mkdir -p models

echo "  → Aya Expanse 32B 4-bit (~18GB)"
huggingface-cli download mlx-community/aya-expanse-32b-4bit \
  --local-dir ./models/aya-expanse-32b-4bit \
  --local-dir-use-symlinks False

echo "  → Llama 3.3 70B Instruct 4-bit (~40GB)"
huggingface-cli download mlx-community/Meta-Llama-3.3-70B-Instruct-4bit \
  --local-dir ./models/Meta-Llama-3.3-70B-Instruct-4bit \
  --local-dir-use-symlinks False

# 4. Prepare MLX training data (so it's ready when we fine-tune)
echo "[4/4] Preparing training data..."
PYTHONPATH=. python scripts/train_mlx.py --prepare-only

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. bash baseline.sh   # 2-3h: measure 6 baselines before fine-tuning"
echo "  2. bash train.sh      # only after baselines reveal the best setup"
echo "  3. bash infer.sh      # generate final submission"
