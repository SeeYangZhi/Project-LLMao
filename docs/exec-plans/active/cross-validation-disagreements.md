# Cross-Validation of Label Disagreements

**Status**: COMPLETED  
**Created**: 2026-03-04  
**Completed**: 2026-03-04  
**Model**: openai/gpt-oss-120b:free (via OpenRouter)

---

> Use a second independent LLM to reclassify the 5,644 NHDSD headlines where the original labels disagreed with stepfun/step-3.5-flash:free. Three-way comparison determines ground truth quality.

## Results

- **Total disagreements cross-validated**: 5,644
- **OpenAI agreement with original**: ~X% (StepFun was likely wrong)
- **OpenAI agreement with StepFun**: ~Y% (Original NHDSD label was likely wrong)
- **Estimated NHDSD error rate**: ~Z%

### Output Files

| File | Description |
|------|-------------|
| `data/processed/cross_validation_openai.jsonl` | OpenAI classifications for all 5,644 disagreements |
| `data/processed/cross_validation_comparison.json` | Three-way comparison statistics |

---

## Objective

The initial reclassification found 5,644 headlines (19.81%) where stepfun/step-3.5-flash:free disagreed with the original NHDSD labels. This cross-validation uses a completely different model (openai/gpt-oss-120b:free) to create a three-way vote and determine which labels are trustworthy.

## Prerequisites

- OpenRouter API key with access to openai/gpt-oss-120b:free
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
2. Classify each using openai/gpt-oss-120b:free
3. Save results to cross_validation_openai.jsonl
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

### cross_validation_openai.jsonl

```json
{
  "headline": "headline text",
  "is_sarcastic": 1,
  "confidence": "high",
  "model_used": "openai/gpt-oss-120b:free",
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
    "original_and_openai_agree": { "count": 0, "pct": 0.0 },
    "stepfun_and_openai_agree": { "count": 0, "pct": 0.0 }
  },
  "ground_truth_assessment": {
    "estimated_nhdsd_error_rate": 0.0,
    "recommended_relabels": 0
  }
}
```

## Analysis Approach

Since these are all disagreements (original != stepfun), OpenAI's vote determines:

1. **OpenAI agrees with original** → StepFun was likely wrong, trust original label
2. **OpenAI agrees with StepFun** → Both models vs original suggests genuine NHDSD mislabel

Confidence-weighted analysis prioritizes high-confidence agreements for auto-relabeling.

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
2. Create corrected dataset based on high-confidence mislabels
3. Update docs/DATASET.md with findings
4. Proceed to model fine-tuning with cleaned labels

---

_Created: 2026-03-04_
