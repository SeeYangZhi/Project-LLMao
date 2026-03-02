# Architecture

> System overview and domain boundaries for Project LLMao.

## Project Domain

**Sarcasm Detection & Style Transfer**

- Transform sarcastic headlines to non-sarcastic (and vice versa)
- Strategy-aware generation using control codes
- Interpretable output (understand why sarcasm works)

## Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     PROJECT LLMao                          │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ├── raw/              - Original datasets                  │
│  ├── processed/        - Cleaned, paired data               │
│  └── splits/           - Train/val/test splits              │
├─────────────────────────────────────────────────────────────┤
│  Preprocessing                                │
│  ├── strategy_classifier/  - Classify sarcasm strategies   │
│  ├── pair_matcher/         - Match sarcastic/non pairs     │
│  └── text_augmenter/       - LLM-based style transfer       │
├─────────────────────────────────────────────────────────────┤
│  Model Layer                                                │
│  ├── t5/                - T5-base fine-tuning              │
│  ├── gpt2/              - GPT-2 baseline                    │
│  └── bart/               - BART baseline                   │
├─────────────────────────────────────────────────────────────┤
│  Evaluation                                                │
│  ├── metrics/           - BLEU, perplexity                  │
│  ├── classifier_eval/   - Sarcasm detector evaluation      │
│  └── human_eval/        - Human evaluation harness         │
└─────────────────────────────────────────────────────────────┘
```

## Domain Boundaries

### Data Pipeline (Inbound)

- Raw data ingestion from NHDSD, iSarcasmEval, Sarcasm Corpus V2
- Preprocessing: cleaning, LLM generates counterparts directly, strategy annotation
- Output: paired training data with strategy labels

### Model Training (Core)

- Fine-tuning: T5-base (primary), GPT-2, BART baselines
- Control codes: `<sarcasm>`, `<irony>`, `<satire>`, `<understatement>`, `<overstatement>`, `<rhetorical_question>` (from iSarcasm dataset)
- Hyperparameter logging for reproducibility

### Evaluation (Outbound)

- Automatic metrics: BLEU, perplexity
- Classifier-based: feed outputs to sarcasm detector
- Human evaluation: interpretability assessment

## Key Design Decisions

| Decision            | Rationale                                |
| ------------------- | ---------------------------------------- |
| T5 as primary       | Seq2seq natural for input→output tasks   |
| Control codes       | Interpretable, strategy-aware generation |
| Multiple baselines  | Compare seq2seq vs causal vs denoising   |
| Human eval required | Figurative language needs human judgment |

## Reproducibility Requirements

- Log seed for all random operations
- Log all hyperparameters
- Log dataset version (SHA or date)
- Store model checkpoints with config

---

_Last updated: 2026-03-02_
