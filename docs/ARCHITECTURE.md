# Architecture

> System overview and domain boundaries for Project LLMao.

## Project Domain

**Sarcasm Style Transfer**

Given a sarcastic news headline, generate a non-sarcastic equivalent that
preserves the underlying meaning. Strategy-aware models additionally
identify *which* of six sarcastic mechanisms the input uses
(`sarcasm`, `irony`, `satire`, `overstatement`, `understatement`,
`rhetorical_question`) and decompose the rewrite around that.

## Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                      PROJECT LLMao                          │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ├── data/raw/         NHDSD (28,619 headlines)             │
│  ├── data/processed/   Cleaned + cross-validated pairs      │
│  │                     89,688 strategy-annotated records    │
│  ├── data/splits/      sar_to_non / context_enhanced /      │
│  │                     joint_and_ablate_prepared            │
│  └── data/golden/      Hand-labeled 140 samples × 3 models  │
├─────────────────────────────────────────────────────────────┤
│  Synthetic Annotation Layer (LLM-as-annotator)              │
│  ├── StepFun 3.5 Flash  Pair generation, relabel, augment   │
│  └── Nemotron 3 Nano    Cross-validate disagreements        │
├─────────────────────────────────────────────────────────────┤
│  Model Layer (14 trained models)                            │
│  ├── BART family (Yang Zhi, scripts/train.py)               │
│  │   ├── BART-Base       SFT on sar_to_non                  │
│  │   ├── BART-CE         SFT on context-enhanced split      │
│  │   ├── BART-RL         SFT + REINFORCE + KL penalty       │
│  │   └── BART-CE+RL      BART-CE + REINFORCE + KL           │
│  ├── T5 family (Camille's separate repo)                    │
│  │   ├── T5-Joint        T5-base, joint task prefix         │
│  │   ├── T5-Control      T5-base, plain prefix              │
│  │   └── T5-Joint(small) T5-small joint variant             │
│  ├── LLaMA (Yang Zhi, train_llama*.py)                      │
│  │   ├── LLaMA 3.2 1B    LoRA, headline-only                │
│  │   └── LLaMA Context   LoRA, headline + article body      │
│  └── Ablations (Camille, T5-base × 6)                       │
│      └── ablation_without_{subtype}                         │
│          one of six subtypes held out from training         │
├─────────────────────────────────────────────────────────────┤
│  Evaluation Layer                                           │
│  ├── 7-metric pipeline   flip rate, similarity, perplexity, │
│  │                       BLEU, edit distance, paraphrase,   │
│  │                       LLM-as-judge                       │
│  ├── 3 sarcasm classifiers                                  │
│  │   ├── RoBERTa-Twitter cardiffnlp/twitter-roberta-irony   │
│  │   ├── Bert-Kaggle     helinivan/english-sarcasm-detector │
│  │   └── RoBERTa-News    jkhan447/sarcasm-detection-RoBerta │
│  └── Human eval          140 samples × 3 models × 2         │
│                          annotators, κ > 0.8                │
├─────────────────────────────────────────────────────────────┤
│  Webapp Layer                                               │
│  ├── FastAPI backend     Live mode (BART-CE+RL, LLaMA       │
│  │                       via LMStudio)                      │
│  └── Next.js frontend    Static export to Vercel; 9 pages   │
│                          including dashboard, explorer,     │
│                          mislabels, eval, training, human   │
│                          eval, playground                   │
└─────────────────────────────────────────────────────────────┘
```

## Domain Boundaries

### Data Pipeline (Inbound)

- **Raw ingestion**: NHDSD (TheOnion + HuffPost), iSarcasmEval taxonomy
- **Cleaning**: dedup, whitespace normalization → 28,497 unique headlines
- **Relabel**: StepFun 3.5 Flash classifies each headline; agreement with
  original NHDSD labels is 80.19%
- **Cross-validate**: 5,644 disagreements re-judged by Nemotron 3 Nano;
  72.2% confirmed as NHDSD mislabels (4,076 records flagged in the webapp's
  `/mislabels` page with article links for inspection)
- **Generate pairs**: StepFun produces opposite-style counterparts for
  every cleaned headline (28,536 raw pairs)
- **Strategy augmentation**: 5 additional strategy variants per non→sar
  pair → 89,688 strategy-annotated records
- **Split**: source-level 80/10/10 to prevent leakage

### Model Training (Core)

Two independent training pipelines:

1. **BART + LLaMA** (Yang Zhi, this repo, `scripts/train.py`,
   `train_llama*.py`, `train_rl.py`)
2. **T5 family + ablations** (Camille,
   [`CS4248-project-AY2526S2`](https://github.com/camille-readbean/CS4248-project-AY2526S2),
   `scripts/finetune_T5.py` + SLURM orchestration)

Both feed into the same downstream eval pipeline.

### Evaluation (Outbound)

- **Automated**: 7 metrics × 3 classifiers × 14 models on the 2,857-sample
  test split (`scripts/eval_pipeline.py --multi_classifier`,
  `scripts/batch_eval.py`)
- **Human**: 2 annotators × 140 stratified samples × 3 models on golden
  data (`data/golden/cleaned/`, analysed by
  `scripts/analyze_golden_results.py`)
- **Findings**: see `docs/EVALUATION.md` and the webapp `/eval` and
  `/human-eval` pages

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Strategy-prefix joint task | Forces decomposition before generation — T5-Joint wins meaning preservation (16.4% change vs T5-Control's 25%) |
| Multi-classifier audit | Single classifier flip rate is unreliable: same model shows up to 33pp spread across classifiers |
| Human evaluation as ground truth | Classifiers all fail vs human (κ −0.11 to +0.18), human inter-annotator κ > 0.8 |
| LLM as data annotator, small models for the task | LLM provides supervision that doesn't exist in the wild; small models give inspectable, controllable, fast inference |
| Two parallel training repos | Allows independent iteration; both consume the same data prep |
| 6-way subtype ablation | Tests whether any single sarcasm subtype is load-bearing (finding: no — they cluster within 0.005 similarity) |

## Reproducibility Requirements

- All seeds logged (default 42)
- All hyperparameters in saved `training_config.json` per run
- Dataset SHA / split metadata in `data/splits/split_metadata.json`
- Model checkpoints stored under `outputs/{model}/{direction}/final/`
- LLaMA SFT exports a 50MB LoRA adapter + a merged + GGUF for LMStudio

## Related Documentation

- [DATASET.md](DATASET.md) — data sources, schemas, preprocessing pipeline
- [METHODS.md](METHODS.md) — training recipes per model family
- [EVALUATION.md](EVALUATION.md) — 7-metric pipeline + human evaluation
- [EXPERIMENTS.md](EXPERIMENTS.md) — per-model results
- [PROJECT.md](PROJECT.md) — top-level project description

---

_Last updated: 2026-04-14_
