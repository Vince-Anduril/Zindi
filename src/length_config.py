# Median answer lengths from training data — calibrate generation to these targets
# ROUGE F1 drops sharply if generated length deviates from reference length
MEDIAN_LENGTH_BY_SUBSET = {
    "Aka_Gha": 100,
    "Amh_Eth": 19,
    "Eng_Eth": 24,
    "Eng_Gha": 70,
    "Eng_Ken": 64,
    "Eng_Uga": 73,
    "Lug_Uga": 68,
    "Swa_Ken": 66,
}

# Allow ±30% around median for min/max tokens
def get_token_bounds(subset: str, factor: float = 1.5) -> tuple[int, int]:
    median = MEDIAN_LENGTH_BY_SUBSET.get(subset, 70)
    # Words → tokens approx ×1.3 for African languages (subword tokenization)
    tokens = int(median * 1.3)
    min_tokens = max(10, int(tokens / factor))
    max_tokens = int(tokens * factor)
    return min_tokens, max_tokens
