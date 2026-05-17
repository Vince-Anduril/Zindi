"""
Inference with fine-tuned Aya Expanse 8B on Mac Studio.

Usage:
  python scripts/infer_mlx.py
  python scripts/infer_mlx.py --model CohereForAI/aya-expanse-8b  # base model (no fine-tune)
"""
import sys
import argparse
import pandas as pd  # type: ignore
from pathlib import Path
from tqdm import tqdm  # type: ignore

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import LANGUAGE_MAP, OUTPUT_DIR
from src.data import load_test
from src.rag import get_context

MODEL_ID = "CohereForAI/aya-expanse-8b"
ADAPTER_PATH = ROOT / "outputs" / "aya-expanse-8b-finetuned"


def build_prompt(question: str, subset: str, context: dict | None) -> str:
    lang = LANGUAGE_MAP.get(subset, subset)
    if context:
        return (
            f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
            f"Language: {lang}\n\n"
            f"Related example on the same topic:\n"
            f"Q: {context['question']}\nA: {context['answer']}\n\n"
            f"Answer this health question in {lang} using similar vocabulary:\n"
            f"Q: {question}"
            f"<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        )
    else:
        return (
            f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
            f"Language: {lang}\n\nAnswer this health question in {lang}:\nQ: {question}"
            f"<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter", default=str(ADAPTER_PATH))
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "submission_mlx.csv"))
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    from mlx_lm import load, generate

    # Resolve to absolute path — mlx-lm rejects relative paths
    model_path = str(Path(args.model).resolve())
    print(f"Loading model: {model_path}")
    if not args.no_adapter and Path(args.adapter).exists():
        adapter_path = str(Path(args.adapter).resolve())
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        print(f"Loaded adapter from {adapter_path}")
    else:
        model, tokenizer = load(model_path)
        print("Loaded base model (no adapter)")

    test = load_test()
    rows = []

    for _, row in tqdm(test.iterrows(), total=len(test), desc="Generating"):
        ctx = get_context(row["ID"])
        prompt = build_prompt(row["input"], row["subset"], ctx)

        from src.length_config import get_token_bounds
        min_tok, max_tok = get_token_bounds(row["subset"])
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tok,
            verbose=False,
        )
        # Strip any trailing turn tokens
        answer = response.split("<|END_OF_TURN_TOKEN|>")[0].strip()
        rows.append({"ID": row["ID"], "TargetRLF1": answer, "TargetR1F1": answer, "TargetLLM": answer})

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nSubmission saved: {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
