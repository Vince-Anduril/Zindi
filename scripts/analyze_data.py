"""
Comprehensive data analysis — runs on M1, produces actionable artifacts:

  outputs/analysis/report.md           — human-readable summary
  outputs/analysis/length_stats.json   — precise length stats per language
  outputs/analysis/ngrams.json         — top n-grams per language
  outputs/analysis/phrases.json        — opening/closing phrases per language
  outputs/analysis/anomalies.json      — quality issues detected
  outputs/analysis/topics.json         — topic clusters (TF-IDF + k-means)

Usage: python scripts/analyze_data.py
"""
import sys
import re
import json
import statistics
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # type: ignore
from src.data import load_train, load_val, load_test, build_train_index

OUT = ROOT / "outputs" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


def tokenize(text: str) -> list[str]:
    """Basic whitespace tokenizer that preserves Unicode (Amharic, Akan, etc.)."""
    return str(text).split()


def char_ngrams(text: str, n: int) -> list[str]:
    s = str(text)
    return [s[i:i+n] for i in range(len(s)-n+1)]


def word_ngrams(text: str, n: int) -> list[str]:
    toks = tokenize(text)
    return [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]


# ============================================================
# 1. Length statistics per language (precise)
# ============================================================
def analyze_lengths(train, val, test):
    print("[1/6] Length statistics...")
    out = {}
    for label, df in [("train", train), ("val", val), ("test", test)]:
        out[label] = {}
        for subset in df["subset"].unique():
            col = "output" if "output" in df.columns else "input"
            lengths = [len(tokenize(t)) for t in df[df["subset"] == subset][col]]
            char_lengths = [len(str(t)) for t in df[df["subset"] == subset][col]]
            if not lengths:
                continue
            sl = sorted(lengths)
            sc = sorted(char_lengths)
            n = len(lengths)
            out[label][subset] = {
                "n": n,
                "words": {
                    "p10": sl[n//10],
                    "p25": sl[n//4],
                    "median": sl[n//2],
                    "p75": sl[3*n//4],
                    "p90": sl[9*n//10],
                    "mean": round(sum(lengths)/n, 1),
                    "std": round(statistics.stdev(lengths) if n>1 else 0, 1),
                },
                "chars": {
                    "median": sc[n//2],
                    "p90": sc[9*n//10],
                },
            }
    (OUT / "length_stats.json").write_text(json.dumps(out, indent=2))
    return out


# ============================================================
# 2. Top n-grams per language (vocab to bias generation toward)
# ============================================================
def analyze_ngrams(train):
    print("[2/6] Top n-grams per language...")
    out = {}
    for subset in train["subset"].unique():
        answers = train[train["subset"] == subset]["output"].tolist()
        unigrams = Counter()
        bigrams = Counter()
        trigrams = Counter()
        for ans in answers:
            toks = tokenize(ans)
            unigrams.update(toks)
            bigrams.update(word_ngrams(ans, 2))
            trigrams.update(word_ngrams(ans, 3))
        # Filter very common short tokens for unigrams
        out[subset] = {
            "top_unigrams": unigrams.most_common(50),
            "top_bigrams": bigrams.most_common(50),
            "top_trigrams": trigrams.most_common(50),
            "vocab_size": len(unigrams),
            "n_answers": len(answers),
            "median_unique_tokens": statistics.median([len(set(tokenize(a))) for a in answers]),
        }
    (OUT / "ngrams.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ============================================================
# 3. Common opening and closing phrases per language
# ============================================================
def analyze_phrases(train):
    print("[3/6] Opening/closing phrases...")
    out = {}
    for subset in train["subset"].unique():
        answers = train[train["subset"] == subset]["output"].tolist()
        openings_3 = Counter()
        openings_5 = Counter()
        closings_3 = Counter()
        closings_5 = Counter()
        for ans in answers:
            toks = tokenize(ans)
            if len(toks) >= 3:
                openings_3[" ".join(toks[:3])] += 1
                closings_3[" ".join(toks[-3:])] += 1
            if len(toks) >= 5:
                openings_5[" ".join(toks[:5])] += 1
                closings_5[" ".join(toks[-5:])] += 1
        out[subset] = {
            "top_openings_3w": openings_3.most_common(20),
            "top_openings_5w": openings_5.most_common(20),
            "top_closings_3w": closings_3.most_common(20),
            "top_closings_5w": closings_5.most_common(20),
            "n_unique_openings_5w": len(openings_5),
            "concentration_top10_openings": sum(c for _,c in openings_5.most_common(10)) / max(1, sum(openings_5.values())),
        }
    (OUT / "phrases.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ============================================================
# 4. Question patterns (types, starting words)
# ============================================================
def analyze_questions(train, test):
    print("[4/6] Question patterns...")
    out = {}
    for label, df in [("train", train), ("test", test)]:
        out[label] = {}
        for subset in df["subset"].unique():
            questions = df[df["subset"] == subset]["input"].tolist()
            first_words = Counter()
            lengths = []
            has_qmark = 0
            for q in questions:
                toks = tokenize(q)
                if toks:
                    first_words[toks[0].lower()] += 1
                lengths.append(len(toks))
                if "?" in str(q):
                    has_qmark += 1
            sl = sorted(lengths)
            n = len(lengths)
            out[label][subset] = {
                "n": n,
                "top_first_words": first_words.most_common(15),
                "length_median": sl[n//2] if n else 0,
                "length_p90": sl[9*n//10] if n else 0,
                "has_question_mark": has_qmark,
                "pct_question_mark": round(100*has_qmark/n, 1) if n else 0,
            }
    (OUT / "questions.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ============================================================
# 5. Quality / anomalies detection
# ============================================================
def analyze_quality(train, val, test):
    print("[5/6] Quality and anomalies...")
    anomalies = {"train": {}, "val": {}, "test": {}}

    for label, df in [("train", train), ("val", val), ("test", test)]:
        a = anomalies[label]

        # Missing or empty fields
        a["n_total"] = len(df)
        a["null_input"] = int(df["input"].isnull().sum())
        a["empty_input"] = int((df["input"].fillna("").str.strip() == "").sum())
        if "output" in df.columns:
            a["null_output"] = int(df["output"].isnull().sum())
            a["empty_output"] = int((df["output"].fillna("").str.strip() == "").sum())

        # Duplicates
        a["duplicate_questions"] = int(df["input"].duplicated().sum())
        a["duplicate_ids"] = int(df["ID"].duplicated().sum())

        # Very short / very long answers (potential noise)
        if "output" in df.columns:
            lens = df["output"].fillna("").apply(lambda x: len(str(x).split()))
            a["answer_under_5_words"] = int((lens < 5).sum())
            a["answer_under_2_words"] = int((lens < 2).sum())
            a["answer_over_500_words"] = int((lens > 500).sum())

        # Suspect encoding (mojibake markers)
        if "output" in df.columns:
            mojibake = df["output"].fillna("").apply(lambda x: any(m in str(x) for m in ["�", "Ã©", "Ã¨", "â€"]))
            a["mojibake_count"] = int(mojibake.sum())

        # Language consistency: detect Latin alphabet in non-Latin subsets (Amharic)
        if label == "train":
            for subset in ["Amh_Eth"]:
                sub_df = df[df["subset"] == subset]
                if len(sub_df) > 0 and "output" in df.columns:
                    has_latin = sub_df["output"].apply(
                        lambda x: bool(re.search(r'[a-zA-Z]', str(x))) if pd.notna(x) else False
                    )
                    a[f"{subset}_has_latin_chars"] = int(has_latin.sum())

        # Code-switching detection: English words in African-language answers
        english_markers = {"the", "and", "is", "are", "for", "with", "you", "this", "that"}
        if "output" in df.columns:
            for subset in ["Aka_Gha", "Lug_Uga", "Swa_Ken", "Amh_Eth"]:
                sub_df = df[df["subset"] == subset]
                if len(sub_df) == 0:
                    continue
                def has_english(text):
                    words = set(str(text).lower().split())
                    return len(words & english_markers) >= 2  # at least 2 English markers
                count = sub_df["output"].apply(has_english).sum()
                a[f"{subset}_likely_codeswitched"] = int(count)

    (OUT / "anomalies.json").write_text(json.dumps(anomalies, indent=2))
    return anomalies


# ============================================================
# 6. Generate the markdown report
# ============================================================
def generate_report(lengths, ngrams, phrases, questions, anomalies):
    print("[6/6] Generating report...")
    lines = []
    lines.append("# Data Analysis Report\n")
    lines.append("Generated for Zindi Multilingual Health QA Challenge\n")
    lines.append("\n---\n")

    # 1. Overview
    lines.append("## 1. Dataset Overview\n")
    lines.append(f"- Train: {anomalies['train']['n_total']:,} rows")
    lines.append(f"- Val:   {anomalies['val']['n_total']:,} rows")
    lines.append(f"- Test:  {anomalies['test']['n_total']:,} rows")
    lines.append("")

    # 2. Critical length stats
    lines.append("## 2. Answer Length per Language — USE THESE FOR LENGTH CALIBRATION\n")
    lines.append("| Lang | p10 | p25 | **median** | p75 | p90 | mean | std | n |")
    lines.append("|------|-----|-----|------------|-----|-----|------|-----|---|")
    train_lens = lengths.get("train", {})
    for subset in sorted(train_lens.keys()):
        s = train_lens[subset]["words"]
        n = train_lens[subset]["n"]
        lines.append(f"| {subset} | {s['p10']} | {s['p25']} | **{s['median']}** | {s['p75']} | {s['p90']} | {s['mean']} | {s['std']} | {n} |")
    lines.append("")

    # Check train/val/test length consistency
    lines.append("### Train vs Val length consistency\n")
    val_lens = lengths.get("val", {})
    lines.append("| Lang | train_median | val_median | drift |")
    lines.append("|------|--------------|------------|-------|")
    for subset in sorted(train_lens.keys()):
        tm = train_lens[subset]["words"]["median"]
        vm = val_lens.get(subset, {}).get("words", {}).get("median", "N/A")
        if vm != "N/A":
            drift = abs(tm - vm)
            warn = " ⚠️" if drift > 5 else ""
            lines.append(f"| {subset} | {tm} | {vm} | {drift}{warn} |")
    lines.append("")

    # 3. Top opening phrases - critical for prompt engineering
    lines.append("## 3. Top 5 Opening Phrases per Language\n")
    lines.append("If the model starts answers with these, ROUGE goes up.\n")
    for subset in sorted(phrases.keys()):
        lines.append(f"\n### {subset}\n")
        lines.append("| Phrase | Count |")
        lines.append("|--------|-------|")
        for phrase, count in phrases[subset]["top_openings_5w"][:5]:
            lines.append(f"| `{phrase}` | {count} |")
        conc = phrases[subset]["concentration_top10_openings"]
        lines.append(f"\n→ Top-10 openings cover **{conc*100:.1f}%** of answers")
    lines.append("")

    # 4. Top closing phrases
    lines.append("## 4. Top 5 Closing Phrases per Language\n")
    for subset in sorted(phrases.keys()):
        lines.append(f"\n### {subset}\n")
        lines.append("| Phrase | Count |")
        lines.append("|--------|-------|")
        for phrase, count in phrases[subset]["top_closings_5w"][:5]:
            lines.append(f"| `{phrase}` | {count} |")
    lines.append("")

    # 5. Top trigrams (vocabulary to reproduce)
    lines.append("## 5. Top 10 Trigrams per Language (key vocabulary)\n")
    for subset in sorted(ngrams.keys()):
        lines.append(f"\n### {subset}\n")
        for tg, c in ngrams[subset]["top_trigrams"][:10]:
            lines.append(f"- `{tg}` ({c})")
    lines.append("")

    # 6. Question patterns
    lines.append("## 6. Question Patterns\n")
    lines.append("### Train vs Test first-word distribution\n")
    qt = questions.get("train", {})
    qe = questions.get("test", {})
    for subset in sorted(qt.keys()):
        lines.append(f"\n#### {subset}")
        train_words = dict(qt[subset]["top_first_words"][:5])
        test_words = dict(qe.get(subset, {}).get("top_first_words", [])[:5]) if subset in qe else {}
        lines.append(f"- Train top first words: {list(train_words.items())[:5]}")
        lines.append(f"- Test top first words:  {list(test_words.items())[:5]}")
    lines.append("")

    # 7. Anomalies
    lines.append("## 7. Anomalies and Quality Issues\n")
    for label in ["train", "val", "test"]:
        a = anomalies[label]
        lines.append(f"\n### {label}")
        for k, v in sorted(a.items()):
            if isinstance(v, int) and v > 0 and k != "n_total":
                lines.append(f"- {k}: **{v}**")
    lines.append("")

    # 8. Actionable conclusions
    lines.append("## 8. ACTIONABLE CONCLUSIONS\n")
    lines.append("Generated automatically from above stats.\n")
    actions = []

    # Length calibration
    actions.append("### Length calibration (CRITICAL)\n")
    actions.append("Set max_tokens at inference per subset:")
    actions.append("```python")
    actions.append("MAX_TOKENS_BY_SUBSET = {")
    for subset in sorted(train_lens.keys()):
        p90 = train_lens[subset]["words"]["p90"]
        actions.append(f'    "{subset}": {int(p90 * 1.3)},  # p90 words * 1.3 (tok/word)')
    actions.append("}")
    actions.append("```")
    actions.append("")

    # Codeswitching warning
    for subset in ["Aka_Gha", "Lug_Uga", "Swa_Ken", "Amh_Eth"]:
        cs = anomalies["train"].get(f"{subset}_likely_codeswitched", 0)
        total = anomalies["train"]["n_total"]
        if cs > total * 0.05:
            actions.append(f"⚠️ **{subset}**: {cs} answers may contain English code-switching")
    actions.append("")

    lines.extend(actions)

    report = "\n".join(lines)
    (OUT / "report.md").write_text(report)
    return report


def main():
    train = load_train()
    val = load_val()
    test = load_test()

    print(f"Loaded: train={len(train)}, val={len(val)}, test={len(test)}\n")

    lengths = analyze_lengths(train, val, test)
    ngrams = analyze_ngrams(train)
    phrases = analyze_phrases(train)
    questions = analyze_questions(train, test)
    anomalies = analyze_quality(train, val, test)
    generate_report(lengths, ngrams, phrases, questions, anomalies)

    print(f"\n✅ Analysis complete. Reports written to:")
    print(f"  {OUT}/report.md          ← read this first")
    print(f"  {OUT}/length_stats.json")
    print(f"  {OUT}/ngrams.json")
    print(f"  {OUT}/phrases.json")
    print(f"  {OUT}/questions.json")
    print(f"  {OUT}/anomalies.json")


if __name__ == "__main__":
    main()
