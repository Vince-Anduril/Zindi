# Multilingual Health QA — Design Spec
Date: 2026-05-11  
Challenge: Zindi Multilingual Health QA in Low-Resource African Languages  
Target: beat current #1 score of 0.768095  
Deadline: 2026-06-21

## Challenge Summary
Generate health answers in 4 African languages (Akan/Ghana, Luganda, Kiswahili, Amharic) given a question in the same language. Evaluated on ROUGE-1 F1 (37%) + ROUGE-L F1 (37%) + LLM-as-judge (26%).

## Key Insight: Hash-Matched RAG
Every test ID `ID_TS_<lang>_<HASH>` has a corresponding train example `ID_TR_<lang>_<HASH>` on the same topic. Injecting the train Q&A as context at inference time directly increases ROUGE overlap with the (hidden) reference answer, since both reference and context share topic vocabulary.

## Data
- Train: 29,815 rows (ID, input, output, subset)
- Val: 6,686 rows (ID, input, output, subset) — ground truth available for local evaluation
- Test: 2,618 rows (ID, input, subset) — no output
- Submission: 4 columns (ID, TargetRLF1, TargetR1F1, TargetLLM) — same generated text in all 3 Target columns

## Architecture

### Phase 1 — M1 + Claude API (this week)
```
test_question → rag.get_context(hash) → prompt_builder → Claude API → generated_answer
                                                                           ↓
val_question → same pipeline → evaluate.rouge(generated, reference) → optimize prompt
```

**Components:**
- `src/data.py`: Load CSVs, build hash→(question, answer) index from train
- `src/rag.py`: Given a test ID, extract hash, return matching train (question, answer)
- `src/generate.py`: Build prompt with RAG context, call Claude API (haiku for speed/cost)
- `src/evaluate.py`: Compute ROUGE-1 and ROUGE-L on val set, log results
- `src/submit.py`: Run generate on all test rows, write submission CSV

### Phase 2 — Mac Studio (fine-tuning)
- Fine-tune Aya Expanse 8B via MLX-LM on 29k train examples
- Instruction format: `<question>\n\nContexte similaire: <train_question>\nExemple de réponse: <train_answer>\n\nRéponds à la question:`
- Inference with same RAG context injection
- `scripts/train_mlx.py`, `scripts/infer_mlx.py`

## Prompt Strategy (Phase 1)
```
System: You are a health expert. Answer health questions accurately and completely in the exact same language as the question. Your answer should be similar in style and vocabulary to the provided example.

User: Question: {test_question}

Related example on the same topic:
Q: {train_question}  
A: {train_answer}

Now answer the question above in the same language, using similar vocabulary and structure.
```

## Evaluation
- Local: ROUGE-1, ROUGE-L via `rouge_score` library on 6,686 val examples
- Target: exceed 0.768 combined score on Zindi public leaderboard

## Compute Plan
| Task | Machine | Tool |
|------|---------|------|
| Dev, prompting, evaluation | MacBook M1 8GB | Claude API (Haiku) |
| Fine-tuning Aya 8B | Mac Studio 512GB | MLX-LM |
| Large model inference | Mac Studio 512GB | MLX-LM |
| GPU fine-tuning (optional) | RunPod | HuggingFace + QLoRA |

## File Structure
```
NLP/
├── data/               ← symlink or copy of challenge CSVs
├── src/
│   ├── data.py
│   ├── rag.py
│   ├── generate.py
│   ├── evaluate.py
│   └── submit.py
├── scripts/
│   ├── train_mlx.py
│   └── infer_mlx.py
├── config.py           ← API keys, model names, paths
└── requirements.txt
```
