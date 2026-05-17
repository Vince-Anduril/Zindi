#!/bin/bash
# Phase 0 — Run all 6 baseline experiments before any fine-tuning.
# Should take ~1-2h on Mac Studio M4 Ultra.
set -e

if [[ ! -d .venv ]]; then
    echo "ERROR: .venv not found. Run 'bash setup.sh' first."
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== Phase 0: Baseline measurement ==="
echo "Running 6 experiments on 200 stratified val samples..."

PYTHONPATH=. python scripts/baseline.py --all --n 200

echo ""
echo "=== Done. Check outputs/baselines/summary.json ==="
