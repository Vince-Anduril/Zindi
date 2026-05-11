"""
Generation using local MLX model (open-source, competition-compliant).
Falls back to a simple context-copy if no model is loaded.
"""
from pathlib import Path
from config import LANGUAGE_MAP

ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = str(ROOT / "models" / "aya-expanse-8b")
DEFAULT_ADAPTER = str(ROOT / "outputs" / "aya-expanse-8b-finetuned")

# Module-level model cache (load once, reuse)
_model = None
_tokenizer = None


def load_model(model_path: str = DEFAULT_MODEL, adapter_path: str | None = DEFAULT_ADAPTER):
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    from mlx_lm import load  # type: ignore  # Mac Studio only, not installed on M1
    adapter = adapter_path if adapter_path and Path(adapter_path).exists() else None
    print(f"Loading model: {model_path}" + (f" + adapter" if adapter else ""))
    _model, _tokenizer = load(model_path, adapter_path=adapter)
    return _model, _tokenizer


def build_prompt(question: str, subset: str, context: dict | None) -> str:
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

    return (
        f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}"
        f"<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    )


def generate_answer(
    question: str,
    subset: str,
    context: dict | None,
    model=None,
    tokenizer=None,
    max_tokens: int = 512,
) -> str:
    if model is None:
        model, tokenizer = load_model()

    from mlx_lm import generate  # type: ignore  # Mac Studio only
    prompt = build_prompt(question, subset, context)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return response.split("<|END_OF_TURN_TOKEN|>")[0].strip()


def generate_answer_fallback(context: dict | None) -> str:
    """Return training answer directly — useful as ultra-fast baseline."""
    return context["answer"] if context else ""


if __name__ == "__main__":
    from src.data import load_val
    from src.rag import get_context
    from src.evaluate import compute_rouge

    val = load_val()
    row = val.iloc[0]
    ctx = get_context(row["ID"])

    model, tokenizer = load_model()
    answer = generate_answer(row["input"], row["subset"], ctx, model, tokenizer)

    print(f"Generated:\n{answer}\n")
    print(f"Reference:\n{row['output']}\n")
    scores = compute_rouge([answer], [row["output"]])
    print(f"ROUGE-1: {scores['rouge1_f1']:.4f}  ROUGE-L: {scores['rougeL_f1']:.4f}")
