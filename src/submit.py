"""Generate submission CSV for Zindi challenge."""
import anthropic
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, OUTPUT_DIR
from src.data import load_test, extract_subset
from src.rag import get_context
from src.generate import generate_answer


def generate_submission(
    model: str = CLAUDE_MODEL,
    output_path: Path | None = None,
    resume_from: Path | None = None,
) -> pd.DataFrame:
    """
    Generate answers for all test questions and write submission CSV.
    Supports resuming from a partial run.
    """
    test = load_test()
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Load already-generated answers if resuming
    done = {}
    if resume_from and resume_from.exists():
        prev = pd.read_csv(resume_from)
        done = dict(zip(prev["ID"], prev["TargetRLF1"]))
        print(f"Resuming: {len(done)} already done")

    rows = []
    for _, row in tqdm(test.iterrows(), total=len(test), desc="Generating"):
        row_id = row["ID"]

        if row_id in done:
            rows.append({"ID": row_id, "TargetRLF1": done[row_id], "TargetR1F1": done[row_id], "TargetLLM": done[row_id]})
            continue

        ctx = get_context(row_id)
        subset = row["subset"]

        try:
            answer = generate_answer(row["input"], subset, ctx, client=client, model=model)
        except Exception as e:
            print(f"Error on {row_id}: {e}")
            answer = ctx["answer"] if ctx else ""

        rows.append({"ID": row_id, "TargetRLF1": answer, "TargetR1F1": answer, "TargetLLM": answer})

        # Auto-save every 100 rows
        if len(rows) % 100 == 0 and output_path:
            pd.DataFrame(rows).to_csv(output_path, index=False)

    df = pd.DataFrame(rows)

    if output_path is None:
        output_path = OUTPUT_DIR / f"submission_{model.replace('/', '-')}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else CLAUDE_MODEL
    output = OUTPUT_DIR / f"submission_{model.split('-')[1]}.csv"
    generate_submission(model=model, output_path=output, resume_from=output)
