# Dataset Documentation

> Data sources, schemas, and preprocessing for Project LLMao.

## Primary Dataset: NHDSD

**Name**: News Headlines Dataset for Sarcasm Detection
**Source**: Kaggle - https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
**Size**: ~28,000 news headlines
**Type**: Text Classification / Generation

### Schema

```json
{
  "is_sarcastic": 1, // 1 = sarcastic, 0 = non-sarcastic
  "headline": "string", // News headline text
  "article_link": "url" // Original article URL
}
```

### Data Sources

| Source    | Type          | Count      |
| --------- | ------------- | ---------- |
| TheOnion  | Sarcastic     | 13,634     |
| HuffPost  | Non-sarcastic | 14,985     |
| **Total** |               | **28,619** |

### Characteristics

- **Professional writing**: No spelling mistakes, formal language
- **Self-contained**: No reply context required
- **High-quality labels**: TheOnion = intentional sarcasm
- **Topic-matched**: Headlines cover similar events

## Reference Datasets

### iSarcasmEval

**GitHub**: https://github.com/iabufarha/iSarcasmEval
**Purpose**: Additional sarcasm data with strategy annotations

### Sarcasm Corpus V2

**GitHub**: https://github.com/soraby/sarcasm2
**Purpose**: Reference dataset for comparison

## Data Processing Pipeline

### Step 1: Data Cleaning & Reclassification

1. **Clean NHDSD**: Remove duplicates, normalize whitespace → 28,497 headlines
2. **Binary reclassification**: Reclassify all headlines with stepfun/step-3.5-flash:free
   - Agreement with original labels: 80.19%
   - 5,644 disagreements flagged
3. **Cross-validation**: 5,644 disagreements re-evaluated with nvidia/nemotron-3-nano-30b-a3b:free
   - 72.2% of disagreements confirmed as NHDSD mislabels (~4,076 headlines)
4. **Censorship filtering**: 6 headlines blocked by content filter, 3 additional filtered during strategy augmentation

| File | Description | Count |
|------|-------------|-------|
| `data/processed/nhdsd_cleaned.json` | Cleaned NHDSD | 28,497 |
| `data/processed/nhdsd_reclassified.jsonl` | Fresh binary labels | 28,497 |
| `data/processed/label_disagreements.jsonl` | Original vs StepFun disagreements | 5,644 |
| `data/processed/cross_validation_secondary.jsonl` | Cross-validation results | 5,644 |

### Step 2: Sarcasm Pair Generation

Generate opposite-style headlines using stepfun/step-3.5-flash:free (via OpenRouter):

- **Sarcastic → Non-sarcastic**: 13,588 pairs
- **Non-sarcastic → Sarcastic**: 14,948 pairs (after dedup + censorship removal)
- **Total**: 28,536 pairs
- Each generated sarcastic headline is annotated with one of 6 strategies

| File | Description | Count |
|------|-------------|-------|
| `data/processed/sarcasm_pairs_step35_clean.jsonl` | All generated pairs | 28,573 |
| `data/processed/sarcasm_pairs_non_to_sarcastic.jsonl` | Non→Sarcastic pairs | 14,948 |
| `data/processed/sarcasm_pairs_sarcastic_to_non.jsonl` | Sarcastic→Non pairs | 13,588 |

### Step 3: Strategy Augmentation

For each `non_to_sarcastic` pair, generate 5 additional sarcastic variants (one per missing strategy), creating a complete 6-strategy dataset per source headline.

**Model**: stepfun/step-3.5-flash:free (temperature 0.8)

| Metric | Value |
|--------|-------|
| Source headlines | 14,948 |
| Augmented variants | 74,740 |
| Complete sources (6/6 strategies) | 14,948 (100%) |
| Total merged records | 89,688 |

**Strategy distribution (merged dataset)**:

| Strategy | Count |
|----------|-------|
| sarcasm | 14,948 |
| irony | 14,948 |
| satire | 14,948 |
| understatement | 14,948 |
| overstatement | 14,948 |
| rhetorical_question | 14,948 |

| File | Description | Count |
|------|-------------|-------|
| `data/processed/sarcasm_pairs_strategy_augmented.jsonl` | All augmented strategy variants | 74,740 |
| `data/processed/sarcasm_pairs_non_to_sarcastic_complete.jsonl` | Original + augmented (complete) | 89,688 |

### Step 4: Train/Val/Test Split

Stratified by original strategy, split at the **source level** (all 6 variants of a source stay together to prevent data leakage).

| Split | Sources | Records | Ratio |
|-------|---------|---------|-------|
| Train | 11,955 | 71,730 | 80% |
| Val | 1,492 | 8,952 | 10% |
| Test | 1,501 | 9,006 | 10% |

**Seed**: 42

Each split contains equal representation of all 6 strategies.

| File | Description | Count |
|------|-------------|-------|
| `data/splits/train.jsonl` | Training set | 71,730 |
| `data/splits/val.jsonl` | Validation set | 8,952 |
| `data/splits/test.jsonl` | Test set | 9,006 |
| `data/splits/split_metadata.json` | Split metadata & stats | — |

## Sarcasm Strategies

Six strategy categories from the iSarcasm dataset, used as control codes:

| Strategy | Description | Example |
|----------|-------------|---------|
| `<sarcasm>` | Contradicts state of affairs, critical | "Great job on the update, everything is broken now" |
| `<irony>` | Contradicts state of affairs, not critical | "I love waiting in line at the DMV" |
| `<satire>` | Appears to support, contains mockery | "Wow, another meeting that could have been an email" |
| `<understatement>` | Undermines importance | "It's just a minor setback" |
| `<overstatement>` | Obviously exaggerated | "I've told you a million times!" |
| `<rhetorical_question>` | Question contradicting reality | "Isn't it just the best feeling to be ignored?" |

## Record Schema

### Complete dataset record (`sarcasm_pairs_non_to_sarcastic_complete.jsonl`)

```json
{
  "original_headline": "Company releases software update",
  "non_sarcastic_source": "Company releases software update",
  "generated_headline": "Great job on the update, everything works perfectly now",
  "strategy": "sarcasm",
  "existing_strategy": "satire",
  "type": "non_to_sarcastic",
  "variant_type": "original|augmented|patch",
  "model_used": "stepfun/step-3.5-flash:free",
  "article_link": "https://..."
}
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/reclassify_nhdsd_binary.py` | Binary reclassification of NHDSD |
| `scripts/cross_validate_disagreements.py` | Cross-validate label disagreements |
| `scripts/augment_strategy_variants.py` | Generate 5 strategy variants per source |
| `scripts/augment_incomplete_sources.py` | Patch missing strategies for incomplete sources |
| `scripts/merge_augmented_variants.py` | Merge original + augmented into complete dataset |
| `scripts/fix_augmented_variants.py` | Deduplicate augmented variants |
| `scripts/split_pairs_by_type.py` | Split pairs by direction type |
| `scripts/create_train_val_test_splits.py` | Create stratified train/val/test splits |

## Data Versioning

- Use SHA or date in filenames
- Log version in experiment configs
- Document any filtering/cleaning steps

---

_Last updated: 2026-03-05_
