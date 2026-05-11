"""
Prepare data + fine-tune Aya Expanse 8B via MLX-LM.

Usage:
  python scripts/train_mlx.py               # prepare data + train
  python scripts/train_mlx.py --prepare-only # only prepare data
"""
import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_train, load_val, build_train_index
from config import LANGUAGE_MAP

DATA_DIR = ROOT / "outputs" / "mlx_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def format_example(question: str, subset: str, answer: str, context: dict | None) -> str:
    lang = LANGUAGE_MAP.get(subset, subset)
    if context:
        user = (
            f"Language: {lang}\n\n"
            f"Related example on the same topic:\n"
            f"Q: {context['question']}\nA: {context['answer']}\n\n"
            f"Answer this health question in {lang} using similar vocabulary:\n"
            f"Q: {question}"
        )
    else:
        user = f"Language: {lang}\n\nAnswer this health question in {lang}:\nQ: {question}"

    # Aya Expanse chat template
    return (
        f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}"
        f"<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        f"{answer}<|END_OF_TURN_TOKEN|>"
    )


def prepare_data():
    print("Preparing MLX training data...")
    train = load_train()
    val = load_val()
    index = build_train_index()

    def write_jsonl(df, path, label):
        rows = []
        for _, row in df.iterrows():
            ctx = index.get(row["hash"])
            text = format_example(row["input"], row["subset"], row["output"], ctx)
            rows.append({"text": text})
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {label}: {len(rows)} examples → {path}")

    write_jsonl(train, DATA_DIR / "train.jsonl", "train")
    write_jsonl(val.head(500), DATA_DIR / "valid.jsonl", "valid")
    print("Data ready.")


def run_training(model_path: str):
    import subprocess
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_path,
        "--train",
        "--data", str(DATA_DIR),
        "--iters", "3000",
        "--batch-size", "4",
        "--lora-layers", "16",
        "--learning-rate", "1e-5",
        "--adapter-path", str(ROOT / "outputs" / "aya-expanse-8b-finetuned"),
        "--val-batches", "25",
        "--save-every", "500",
        "--grad-checkpoint",
    ]
    print("Starting MLX LoRA training...")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model", default=str(ROOT / "models" / "aya-expanse-8b"))
    args = parser.parse_args()

    prepare_data()
    if not args.prepare_only:
        run_training(args.model)
