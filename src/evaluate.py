import json
from pathlib import Path
from rouge_score import rouge_scorer
import pandas as pd
from tqdm import tqdm
from config import OUTPUT_DIR


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1 and ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1_scores, rl_scores = [], []

    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            r1_scores.append(0.0)
            rl_scores.append(0.0)
            continue
        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1_f1": sum(r1_scores) / len(r1_scores),
        "rougeL_f1": sum(rl_scores) / len(rl_scores),
        # Weighted score matching Zindi: ROUGE-1 37% + ROUGE-L 37% (LLM-judge excluded locally)
        "weighted_rouge": 0.37 * sum(r1_scores) / len(r1_scores) + 0.37 * sum(rl_scores) / len(rl_scores),
        "n": len(predictions),
    }


def evaluate_on_val(
    generate_fn,
    n_samples: int | None = None,
    save_path: Path | None = None,
) -> dict:
    """
    Run full evaluation on val set.
    generate_fn: callable(question, subset, context) → str
    """
    from src.data import load_val
    from src.rag import get_context

    val = load_val()
    if n_samples:
        val = val.head(n_samples)

    predictions, references, ids = [], [], []

    for _, row in tqdm(val.iterrows(), total=len(val), desc="Evaluating"):
        ctx = get_context(row["ID"])
        pred = generate_fn(row["input"], row["subset"], ctx)
        predictions.append(pred)
        references.append(row["output"])
        ids.append(row["ID"])

    scores = compute_rouge(predictions, references)

    if save_path:
        results_df = pd.DataFrame({"ID": ids, "prediction": predictions, "reference": references})
        results_df.to_csv(save_path, index=False)
        scores_path = save_path.with_suffix(".json")
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"Saved to {save_path}")

    return scores


if __name__ == "__main__":
    from src.generate import generate_answer
    from src.rag import get_context

    def gen_fn(question, subset, context):
        return generate_answer(question, subset, context)

    # Quick eval on 50 samples
    scores = evaluate_on_val(
        gen_fn,
        n_samples=50,
        save_path=OUTPUT_DIR / "val_eval_sample.csv",
    )
    print("\nScores (50 samples):")
    print(f"  ROUGE-1:  {scores['rouge1_f1']:.4f}")
    print(f"  ROUGE-L:  {scores['rougeL_f1']:.4f}")
    print(f"  Weighted: {scores['weighted_rouge']:.4f}")
