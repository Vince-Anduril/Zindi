#!/bin/bash
# Run once on Mac Studio after SSH connection
# Usage: bash setup.sh

set -e
echo "=== Setup Zindi NLP Challenge ==="

# 1. Install dependencies
echo "[1/3] Installing Python packages..."
pip install mlx-lm huggingface_hub rouge-score pandas tqdm -q

# 2. Download Aya Expanse 8B
echo "[2/3] Downloading Aya Expanse 8B (~16GB)..."
huggingface-cli download CohereForAI/aya-expanse-8b \
  --local-dir ./models/aya-expanse-8b \
  --local-dir-use-symlinks False

# 3. Prepare MLX training data
echo "[3/3] Preparing training data..."
PYTHONPATH=. python scripts/train_mlx.py --prepare-only

echo ""
echo "=== Setup complete! ==="
echo "Next: bash train.sh"
