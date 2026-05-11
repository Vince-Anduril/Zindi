#!/bin/bash
# Phase 0 — Run all 6 baseline experiments before any fine-tuning.
# Should take ~2-3h on Mac Studio M2 Ultra.
# Run AFTER setup.sh has installed mlx-lm and downloaded models.
set -e

echo "=== Phase 0: Baseline measurement ==="
echo "Running 6 experiments on 200 stratified val samples..."

PYTHONPATH=. python scripts/baseline.py --all --n 200

echo ""
echo "=== Done. Check outputs/baselines/summary.json ==="
echo "Next step depends on results. The best experiment tells us:"
echo "  - Which model to fine-tune (Aya 32B or Llama 70B)"
echo "  - Whether RAG context helps"
echo "  - Whether length calibration matters"
