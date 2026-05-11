#!/usr/bin/env python
"""
Main entry point. Usage:

  # Test sur 1 exemple
  python run.py test

  # Évaluation ROUGE sur N exemples du val set
  python run.py eval --n 100

  # Évaluation complète val set
  python run.py eval

  # Générer la soumission complète
  python run.py submit

  # Modèle alternatif
  python run.py submit --model claude-sonnet-4-6
"""
import argparse
import sys
import os
from pathlib import Path

PYTHON = sys.executable
sys.path.insert(0, str(Path(__file__).parent))


def check_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)


def cmd_test(args):
    check_api_key()
    from src.data import load_val
    from src.rag import get_context
    from src.generate import generate_answer
    from src.evaluate import compute_rouge

    val = load_val()
    row = val.iloc[0]
    ctx = get_context(row["ID"])

    print(f"ID:       {row['ID']}")
    print(f"Language: {row['subset']}")
    print(f"Question: {row['input'][:200]}")
    print(f"Context:  {'found' if ctx else 'NOT FOUND'}")
    print()

    answer = generate_answer(row["input"], row["subset"], ctx)
    print(f"Generated:\n{answer}")
    print()
    print(f"Reference:\n{row['output']}")
    print()

    scores = compute_rouge([answer], [row["output"]])
    print(f"ROUGE-1: {scores['rouge1_f1']:.4f}")
    print(f"ROUGE-L: {scores['rougeL_f1']:.4f}")


def cmd_eval(args):
    check_api_key()
    from src.generate import generate_answer
    from src.evaluate import evaluate_on_val
    from config import OUTPUT_DIR

    def gen_fn(q, s, ctx):
        return generate_answer(q, s, ctx)

    n = args.n
    label = f"n{n}" if n else "full"
    save_path = OUTPUT_DIR / f"val_eval_{label}.csv"

    print(f"Evaluating on {'all' if not n else n} val examples...")
    scores = evaluate_on_val(gen_fn, n_samples=n, save_path=save_path)

    print(f"\n{'='*40}")
    print(f"ROUGE-1 F1: {scores['rouge1_f1']:.4f}")
    print(f"ROUGE-L F1: {scores['rougeL_f1']:.4f}")
    print(f"Weighted:   {scores['weighted_rouge']:.4f}  (target: >0.768)")
    print(f"{'='*40}")


def cmd_submit(args):
    check_api_key()
    from src.submit import generate_submission
    from config import OUTPUT_DIR

    model = args.model or os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    output = OUTPUT_DIR / f"submission_{model.split('-')[1]}.csv"

    print(f"Generating submission with model: {model}")
    generate_submission(model=model, output_path=output, resume_from=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("test")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--n", type=int, default=None)

    p_sub = sub.add_parser("submit")
    p_sub.add_argument("--model", default=None)

    args = parser.parse_args()

    if args.cmd == "test":
        cmd_test(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "submit":
        cmd_submit(args)
    else:
        parser.print_help()
