# AGENTS.md

> Agent entry point for Project LLMao. This file serves as a table of contents—see docs/ for full details.

## Quick Start

```bash
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Development environment
uv sync
```

## Project Overview

**Task**: Sarcasm style transfer - generate non-sarcastic versions from sarcastic headlines (and vice versa).

**Datasets**:

- NHDSD (main): ~28K news headlines from TheOnion (sarcastic) + HuffPost (non-sarcastic)
- iSarcasmEval (reference): https://github.com/iabufarha/iSarcasmEval
- Sarcasm Corpus V2 (reference): https://github.com/soraby/sarcasm2

## Documentation Structure

```
docs/
├── ARCHITECTURE.md          # System overview & domain boundaries
├── CORE_BELIEFS.md          # Design philosophy & golden principles
├── DATASET.md               # Data sources, schemas, preprocessing
├── METHODS.md               # Approach: strategy classification, fine-tuning
├── EXPERIMENTS.md           # Experiment tracking & results
├── EVALUATION.md            # Metrics: BLEU, perplexity, human eval
├── REPORT_GUIDE.md          # CS4248 final report requirements
└── exec-plans/completed/    # Completed data pipeline execution plans
```

## Key Conventions

### Code Organization

- `notebooks/` - Jupyter notebooks (classification pipeline, analysis)
- `scripts/data_prep/` - Completed data processing pipeline scripts
- `data/raw/` - Original NHDSD dataset
- `data/processed/` - Final generated datasets (intermediate artifacts in `intermediate/`)
- `data/splits/` - Train/val/test splits (stratified, source-level)

### Sarcasm Strategies (from iSarcasm dataset)

- `<sarcasm>` - Contradicts state of affairs, critical towards addressee
- `<irony>` - Contradicts state of affairs, not obviously critical
- `<satire>` - Appears to support, but contains mockery
- `<understatement>` - Undermines importance
- `<overstatement>` - Obviously exaggerated terms
- `<rhetorical_question>` - Question with inference contradicting reality

## Golden Principles

1. **Reproducibility** - Log all hyperparameters, seeds, dataset versions

## Getting Help

- **Architecture questions**: See `docs/ARCHITECTURE.md`
- **Methodology**: See `docs/METHODS.md`
- **Dataset details**: See `docs/DATASET.md`
- **Report writing**: See `docs/REPORT_GUIDE.md` (CS4248 template requirements)

---

_Last updated: 2026-03-02_
