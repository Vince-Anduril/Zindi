#!/bin/bash
# One-shot setup for Mac Studio (fresh install OK).
# Installs Homebrew → Python → venv → MLX → models.
# Idempotent: rerun safely.
set -e

echo "=== Setup Zindi NLP Challenge — Mac Studio ==="

# 1. Homebrew
if ! command -v brew &> /dev/null; then
    echo "[1/6] Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for current shell
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -f /usr/local/bin/brew ]]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
else
    echo "[1/6] Homebrew already installed."
fi

# Ensure brew is on PATH for the rest of this script
if [[ -d /opt/homebrew/bin ]]; then
    export PATH="/opt/homebrew/bin:$PATH"
elif [[ -d /usr/local/bin ]]; then
    export PATH="/usr/local/bin:$PATH"
fi

# 2. Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo "[2/6] Installing Python 3.11 via Homebrew..."
    brew install python@3.11
else
    echo "[2/6] Python 3.11 already installed."
fi

PYTHON_BIN=$(command -v python3.11 || command -v python3)
echo "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version))"

# 3. Virtual environment
if [[ ! -d .venv ]]; then
    echo "[3/6] Creating virtual environment..."
    $PYTHON_BIN -m venv .venv
else
    echo "[3/6] Virtual environment already exists."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# 4. Python packages
echo "[4/6] Installing Python packages..."
pip install --upgrade pip -q
pip install mlx-lm huggingface_hub rouge-score pandas tqdm numpy -q

# 5. Verify MLX
echo "[5/6] Verifying MLX..."
python -c "import mlx.core as mx; print(f'MLX OK. Default device: {mx.default_device()}')"

# 6. Download pre-quantized 4-bit MLX models
echo "[6/6] Downloading models (20-40 min depending on bandwidth)..."
mkdir -p models

echo "  → Aya Expanse 32B 4-bit (~18GB)"
hf download mlx-community/aya-expanse-32b-4bit \
    --local-dir ./models/aya-expanse-32b-4bit

echo "  → Llama 3.3 70B Instruct 4-bit (~40GB)"
hf download mlx-community/Meta-Llama-3.3-70B-Instruct-4bit \
    --local-dir ./models/Meta-Llama-3.3-70B-Instruct-4bit

# Prepare training data
echo "Preparing training data..."
PYTHONPATH=. python scripts/train_mlx.py --prepare-only

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. bash baseline.sh   # 1-2h: measure 6 baselines"
echo "  2. bash train.sh      # 1-3h: fine-tune the winner"
echo "  3. bash infer.sh      # 30 min: generate submission"
