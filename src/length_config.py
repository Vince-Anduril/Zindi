"""
Length calibration per language subset.

Derived from training data analysis (outputs/analysis/length_stats.json).
ROUGE F1 drops when generated length deviates from reference length:
  - too short → recall down
  - too long  → precision down

For each subset we provide:
  - target_median_words: aim generation length toward this
  - max_tokens: hard upper bound for MLX generation
                = (mean + 0.5*std) * 1.3 tokens/word, capped reasonably
"""

# Word-count statistics from training set (validated against val: drift ≤ 5)
STATS_BY_SUBSET = {
    "Aka_Gha": {"p25": 66,  "median": 100, "p75": 139, "p90": 180, "mean": 105.6, "std": 57.8},
    "Amh_Eth": {"p25": 14,  "median": 19,  "p75": 25,  "p90": 30,  "mean": 20.2,  "std": 8.7},
    "Eng_Eth": {"p25": 19,  "median": 24,  "p75": 30,  "p90": 35,  "mean": 24.5,  "std": 8.8},
    "Eng_Gha": {"p25": 51,  "median": 70,  "p75": 93,  "p90": 120, "mean": 75.1,  "std": 35.2},
    "Eng_Ken": {"p25": 42,  "median": 64,  "p75": 109, "p90": 148, "mean": 78.7,  "std": 48.7},
    "Eng_Uga": {"p25": 45,  "median": 73,  "p75": 131, "p90": 184, "mean": 95.4,  "std": 71.2},
    "Lug_Uga": {"p25": 42,  "median": 68,  "p75": 112, "p90": 150, "mean": 79.7,  "std": 51.2},
    "Swa_Ken": {"p25": 42,  "median": 66,  "p75": 117, "p90": 167, "mean": 84.3,  "std": 57.4},
}

# Median answer length per language - target generation length
MEDIAN_WORDS_BY_SUBSET = {k: v["median"] for k, v in STATS_BY_SUBSET.items()}

# Backwards-compat alias
MEDIAN_LENGTH_BY_SUBSET = MEDIAN_WORDS_BY_SUBSET


def get_token_bounds(subset: str, conservative: bool = False) -> tuple[int, int]:
    """
    Return (min_tokens, max_tokens) for MLX generation.
    Tokens-per-word factor ≈ 1.3 for African languages (subword tokenization),
    ≈ 1.1 for English.

    conservative=True targets median (best for ROUGE F1)
    conservative=False targets p75 (safer, no truncation)
    """
    if subset not in STATS_BY_SUBSET:
        return (50, 250)

    s = STATS_BY_SUBSET[subset]
    tpw = 1.1 if subset.startswith("Eng_") else 1.3

    if conservative:
        # Aim tightly around median - maximizes ROUGE F1
        max_tokens = int(s["median"] * tpw * 1.4)
        min_tokens = max(8, int(s["p25"] * tpw))
    else:
        # Allow up to p75 - safer fallback
        max_tokens = int(s["p75"] * tpw * 1.3)
        min_tokens = max(8, int(s["p25"] * tpw * 0.8))

    return (min_tokens, max_tokens)


# Pre-computed reasonable defaults for MLX max_tokens per subset
MAX_TOKENS_BY_SUBSET = {subset: get_token_bounds(subset, conservative=True)[1]
                       for subset in STATS_BY_SUBSET}

if __name__ == "__main__":
    print(f"{'Subset':<12} {'min_tok':>8} {'max_tok':>8} {'target_words':>14}")
    for subset in sorted(STATS_BY_SUBSET):
        mn, mx = get_token_bounds(subset, conservative=True)
        print(f"{subset:<12} {mn:>8} {mx:>8} {STATS_BY_SUBSET[subset]['median']:>14}")
