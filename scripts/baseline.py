"""
Phase 0 — Baseline measurement on val set BEFORE any fine-tuning.

Runs N experiment configurations on a stratified sample of the val set
and reports ROUGE-1 / ROUGE-L per language and overall.

Usage:
  python scripts/baseline.py --exp A           # single experiment
  python scripts/baseline.py --all             # run all experiments
  python scripts/baseline.py --all --n 200     # sample size

Experiments:
  A — Copy train_answer (no model)              [floor / sanity check]
  B — Aya Expanse 32B zero-shot, no RAG         [model only]
  C — Aya Expanse 32B zero-shot, with RAG       [RAG impact]
  D — Llama 3.3 70B zero-shot, no RAG           [bigger model]
  E — Llama 3.3 70B zero-shot, with RAG         [bigger + RAG]
  F — Best of D/E + strict length calibration   [length impact]
"""
import sys
import json
import time
import random
import argparse
import pandas as pd  # type: ignore
from pathlib import Path
from tqdm import tqdm  # type: ignore

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SEED = 42
random.seed(SEED)

from src.data import load_val, build_train_index
from src.evaluate import compute_rouge
from src.length_config import get_token_bounds
from config import LANGUAGE_MAP, OUTPUT_DIR

BASELINE_DIR = OUTPUT_DIR / "baselines"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

# Pre-quantized 4-bit MLX models — use absolute local paths
# (mlx-lm rejects relative paths via HFValidationError)
MODELS = {
    "aya32b":   str(ROOT / "models" / "aya-expanse-32b-8bit"),
    "llama70b": str(ROOT / "models" / "Llama-3.3-70B-Instruct-4bit"),
}

EXPERIMENTS = {
    "A": {"name": "copy_train_answer",          "model": None,        "rag": True,  "length_cal": False},
    "B": {"name": "aya32b_no_rag",              "model": "aya32b",    "rag": False, "length_cal": False},
    "C": {"name": "aya32b_with_rag",            "model": "aya32b",    "rag": True,  "length_cal": False},
    "D": {"name": "llama70b_no_rag",            "model": "llama70b",  "rag": False, "length_cal": False},
    "E": {"name": "llama70b_with_rag",          "model": "llama70b",  "rag": True,  "length_cal": False},
    "F": {"name": "llama70b_rag_length_cal",    "model": "llama70b",  "rag": True,  "length_cal": True},
}


def stratified_sample(df: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """Sample n rows stratified by language subset, matching test distribution."""
    # Test distribution proportions (from data analysis)
    proportions = {
        "Eng_Uga": 0.284, "Aka_Gha": 0.188, "Eng_Gha": 0.188, "Lug_Uga": 0.143,
        "Swa_Ken": 0.087, "Eng_Ken": 0.064, "Amh_Eth": 0.023, "Eng_Eth": 0.023,
    }
    sampled = []
    for subset, prop in proportions.items():
        target = max(2, int(n * prop))
        subset_df = df[df["subset"] == subset]
        if len(subset_df) < target:
            target = len(subset_df)
        sampled.append(subset_df.sample(n=target, random_state=SEED))
    return pd.concat(sampled).reset_index(drop=True)


def build_prompt_aya(question: str, subset: str, context: dict | None) -> str:
    lang = LANGUAGE_MAP.get(subset, subset)
    if context:
        user = (f"Language: {lang}\n\nRelated example on the same topic:\n"
                f"Q: {context['question']}\nA: {context['answer']}\n\n"
                f"Answer this health question in {lang} using similar vocabulary:\n"
                f"Q: {question}")
    else:
        user = f"Language: {lang}\n\nAnswer this health question in {lang}:\nQ: {question}"
    return (f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}"
            f"<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")


def build_prompt_llama(question: str, subset: str, context: dict | None) -> str:
    lang = LANGUAGE_MAP.get(subset, subset)
    if context:
        user = (f"Answer this health question strictly in {lang}, matching the style of the example.\n\n"
                f"Example (same topic):\nQ: {context['question']}\nA: {context['answer']}\n\n"
                f"Now answer in {lang}:\nQ: {question}")
    else:
        user = f"Answer this health question strictly in {lang}.\nQ: {question}"
    return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a health expert. Answer accurately and in the requested language only."
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def clean_response(text: str) -> str:
    """Strip turn tokens and trailing whitespace."""
    for token in ["<|END_OF_TURN_TOKEN|>", "<|eot_id|>", "<|end_of_text|>"]:
        text = text.split(token)[0]
    return text.strip()


def run_experiment(exp_id: str, df: pd.DataFrame, train_idx: dict) -> dict:
    cfg = EXPERIMENTS[exp_id]
    print(f"\n{'='*60}\nExperiment {exp_id}: {cfg['name']}\n{'='*60}")

    predictions = []

    if cfg["model"] is None:
        # Experiment A: copy train_answer
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Exp {exp_id}"):
            ctx = train_idx.get(row["hash"])
            predictions.append(ctx["answer"] if ctx else "")
    else:
        # Load MLX model
        from mlx_lm import load, generate  # type: ignore
        import mlx.core as mx  # type: ignore
        mx.random.seed(SEED)

        model_id = MODELS[cfg["model"]]
        print(f"Loading {model_id}...")
        t0 = time.time()
        model, tokenizer = load(model_id)
        print(f"Loaded in {time.time()-t0:.1f}s")

        is_aya = "aya" in cfg["model"]
        build_prompt = build_prompt_aya if is_aya else build_prompt_llama

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Exp {exp_id}"):
            ctx = train_idx.get(row["hash"]) if cfg["rag"] else None
            prompt = build_prompt(row["input"], row["subset"], ctx)

            if cfg["length_cal"]:
                _, max_tok = get_token_bounds(row["subset"], conservative=True)
            else:
                max_tok = 512

            response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tok, verbose=False)
            predictions.append(clean_response(response))

        # Free memory
        del model, tokenizer

    # Score
    refs = df["output"].tolist()
    overall = compute_rouge(predictions, refs)

    # Per-language scores
    by_lang = {}
    for subset in df["subset"].unique():
        mask = df["subset"] == subset
        preds_l = [p for p, m in zip(predictions, mask) if m]
        refs_l = [r for r, m in zip(refs, mask) if m]
        if preds_l:
            by_lang[subset] = compute_rouge(preds_l, refs_l)

    result = {
        "exp_id": exp_id,
        "name": cfg["name"],
        "config": cfg,
        "overall": overall,
        "by_language": by_lang,
        "n_samples": len(df),
    }

    # Save predictions for inspection
    out_csv = BASELINE_DIR / f"exp_{exp_id}_{cfg['name']}_predictions.csv"
    pd.DataFrame({
        "ID": df["ID"].values,
        "subset": df["subset"].values,
        "question": df["input"].values,
        "reference": refs,
        "prediction": predictions,
    }).to_csv(out_csv, index=False)

    out_json = BASELINE_DIR / f"exp_{exp_id}_{cfg['name']}_scores.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{exp_id} — Overall ROUGE-1: {overall['rouge1_f1']:.4f}  ROUGE-L: {overall['rougeL_f1']:.4f}  Weighted: {overall['weighted_rouge']:.4f}")
    return result


def print_comparison(results: list[dict]):
    print("\n" + "=" * 90)
    print("BASELINE COMPARISON")
    print("=" * 90)
    print(f"{'Exp':<5} {'Name':<32} {'ROUGE-1':>10} {'ROUGE-L':>10} {'Weighted':>10}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: -x["overall"]["weighted_rouge"]):
        print(f"{r['exp_id']:<5} {r['name']:<32} {r['overall']['rouge1_f1']:>10.4f} {r['overall']['rougeL_f1']:>10.4f} {r['overall']['weighted_rouge']:>10.4f}")

    # Per-language for best experiment
    best = max(results, key=lambda x: x["overall"]["weighted_rouge"])
    print(f"\nBest experiment {best['exp_id']} ({best['name']}) — per language:")
    print(f"{'Language':<12} {'ROUGE-1':>10} {'ROUGE-L':>10}  n")
    for subset, scores in sorted(best["by_language"].items()):
        print(f"{subset:<12} {scores['rouge1_f1']:>10.4f} {scores['rougeL_f1']:>10.4f}  {scores['n']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help="Single experiment ID (A-F)")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--n", type=int, default=200, help="Sample size")
    parser.add_argument("--skip", nargs="*", default=[], help="Experiments to skip")
    args = parser.parse_args()

    val = load_val()
    sample = stratified_sample(val, n=args.n)
    train_idx = build_train_index()
    print(f"Sampled {len(sample)} val examples")
    print("Distribution:", sample["subset"].value_counts().to_dict())

    if args.exp:
        results = [run_experiment(args.exp, sample, train_idx)]
    elif args.all:
        results = []
        for exp_id in EXPERIMENTS:
            if exp_id in args.skip:
                continue
            try:
                results.append(run_experiment(exp_id, sample, train_idx))
            except Exception as e:
                print(f"Experiment {exp_id} FAILED: {e}")
                import traceback; traceback.print_exc()
    else:
        parser.print_help()
        return

    # Final comparison
    if len(results) > 1:
        print_comparison(results)
        with open(BASELINE_DIR / "summary.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nFull results saved to {BASELINE_DIR}/")


if __name__ == "__main__":
    main()
