# Cross-Validation of Label Disagreements

**Status**: COMPLETED  
**Created**: 2026-03-04  
**Completed**: 2026-03-04  
**Secondary Model**: nvidia/nemotron-3-nano-30b-a3b:free (via OpenRouter)

---

> Use a second independent LLM to reclassify the 5,644 NHDSD headlines where the original labels disagreed with stepfun/step-3.5-flash:free. Three-way comparison determines ground truth quality.

## Results

- **Total disagreements cross-validated**: 5,644
- **Secondary model agrees with Original**: 1,568 (27.8%) - StepFun was likely wrong
- **Secondary model agrees with StepFun**: 4,076 (72.2%) - Original NHDSD label was likely wrong
- **Estimated NHDSD error rate**: 72.2% of disagreements

### Key Findings

| Metric | Value |
|--------|-------|
| Originally Sarcastic (3,560) | 509 kept, 3,051 reclassified |
| Originally Non-Sarcastic (2,084) | 1,059 kept, 1,025 reclassified |
| TheOnion disagreements | 3,561 |
| HuffPost disagreements | 2,083 |
| High confidence (both models) | 2,148 |
| Mixed/lower confidence | 3,496 |

### Output Files

| File | Description |
|------|-------------|
| `data/processed/cross_validation_secondary.jsonl` | Secondary model classifications for all 5,644 disagreements |
| `data/processed/cross_validation_comparison.json` | Three-way comparison statistics |

---

## Objective

The initial reclassification found 5,644 headlines (19.81%) where stepfun/step-3.5-flash:free disagreed with the original NHDSD labels. This cross-validation uses a completely different model (nvidia/nemotron-3-nano-30b-a3b:free) to create a three-way vote and determine which labels are trustworthy.

## Prerequisites

- OpenRouter API key with access to nvidia/nemotron-3-nano-30b-a3b:free
- Input file: `data/processed/label_disagreements.jsonl` (5,644 records)

```bash
# Verify API key
echo $OPENROUTER_API_KEY

# Verify input file
wc -l data/processed/label_disagreements.jsonl
```

## Execution Steps

### Step 1: Run Cross-Validation

```bash
uv run python scripts/cross_validate_disagreements.py
```

This will:
1. Load the 5,644 disagreed headlines
2. Classify each using nvidia/nemotron-3-nano-30b-a3b:free
3. Save results to cross_validation_secondary.jsonl
4. Support resume if interrupted

### Step 2: Run Analysis

```bash
uv run python scripts/analyze_cross_validation.py
```

Generates three-way comparison statistics.

## Parallelization Strategy

- **Batch size**: 10 headlines per API request
- **Workers**: 5 parallel threads
- **Rate limit**: 20 req/min (conservative for free tier)
- **Estimated time**: ~30-40 minutes

## Output Schema

### cross_validation_secondary.jsonl

```json
{
  "headline": "headline text",
  "is_sarcastic": 1,
  "confidence": "high",
  "model_used": "nvidia/nemotron-3-nano-30b-a3b:free",
  "article_link": "https://...",
  "original_label": 1,
  "stepfun_label": 0,
  "stepfun_confidence": "medium"
}
```

### cross_validation_comparison.json

```json
{
  "metadata": { "total_disagreements": 5644, ... },
  "vote_outcomes": {
    "original_and_secondary_agree": { "count": 1568, "pct": 27.78 },
    "stepfun_and_secondary_agree": { "count": 4076, "pct": 72.22 }
  },
  "ground_truth_assessment": {
    "estimated_nhdsd_error_rate": 72.22,
    "estimated_mislabels": 4076
  }
}
```

## Analysis Approach

Since these are all disagreements (original != stepfun), the secondary model's vote determines:

1. **Secondary agrees with original** → StepFun was likely wrong, trust original label (27.8% of cases)
2. **Secondary agrees with StepFun** → Both models vs original suggests genuine NHDSD mislabel (72.2% of cases)

**Key insight**: 72.2% of disagreements are likely true NHDSD errors, suggesting ~4,076 mislabeled headlines in the original dataset.

## Time Estimates

| Phase | Duration |
|-------|----------|
| Cross-validation run | ~30-40 min |
| Analysis | <1 min |
| **Total** | **~40 min** |

## Troubleshooting

### Resume After Interruption
```bash
# Simply re-run - script skips already processed
uv run python scripts/cross_validate_disagreements.py
```

### Rate Limit Issues
Edit script to reduce `RATE_LIMIT_PER_MINUTE` if hitting 429 errors.

## Next Steps

1. Review comparison statistics
2. Create corrected dataset: use StepFun labels where both models agree (4,076 headlines)
3. Flag ambiguous cases for human review (2,148 high-confidence disagreements)
4. Update docs/DATASET.md with findings
5. Proceed to model fine-tuning with cleaned labels

---

_Created: 2026-03-04_
