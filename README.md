# Zindi Multilingual Health QA

Target: beat #1 score of 0.768095 (ROUGE-1 37% + ROUGE-L 37% + LLM-judge 26%)

## Setup

```bash
pip install anthropic pandas rouge-score tqdm
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Utilisation

```bash
# Test rapide sur 1 exemple
python run.py test

# Évaluation ROUGE sur 100 exemples du val set
python run.py eval --n 100

# Évaluation complète (6686 exemples, coûteux en API)
python run.py eval

# Générer soumission complète (2618 questions test)
python run.py submit

# Avec Claude Sonnet (meilleure qualité, plus cher)
python run.py submit --model claude-sonnet-4-6
```

## Mac Studio — Fine-tuning

```bash
pip install mlx-lm huggingface_hub
huggingface-cli download CohereForAI/aya-expanse-8b

# Fine-tuning (~2000 steps, ~2-3h sur M2 Ultra)
python scripts/train_mlx.py

# Inférence avec modèle fine-tuné
python scripts/infer_mlx.py

# Inférence modèle de base (sans fine-tune)
python scripts/infer_mlx.py --no-adapter
```

## Structure

```
src/
  data.py      ← parsing CSV + index hash→(question, réponse)
  rag.py       ← trouve l'exemple train par hash topic
  generate.py  ← génération via Claude API avec contexte RAG
  evaluate.py  ← ROUGE-1 + ROUGE-L sur val set
  submit.py    ← génère le CSV de soumission Zindi

scripts/
  train_mlx.py   ← fine-tuning Aya Expanse 8B (Mac Studio)
  infer_mlx.py   ← inférence MLX (Mac Studio)

outputs/           ← fichiers générés (soumissions, évals)
```

## Stratégie

Pour chaque question test `ID_TS_<lang>_<HASH>`, on retrouve l'exemple
d'entraînement `ID_TR_<lang>_<HASH>` (même sujet, même hash).
On l'injecte comme contexte dans le prompt → l'overlap ROUGE avec la
réponse de référence (cachée) monte car les deux parlent du même sujet.
