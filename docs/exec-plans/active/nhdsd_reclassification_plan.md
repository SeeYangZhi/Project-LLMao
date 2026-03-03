# NHDSD Binary Reclassification Plan

> Reclassify the NHDSD dataset using binary classification (sarcastic vs non-sarcastic) with stepfun/step-3.5-flash:free model. Distrusts original labels.

## Objective

Clean and reclassify ALL ~28,619 headlines from the NHDSD dataset with fresh binary labels (is_sarcastic: 0 or 1) using the stepfun 3.5 model. The original NHDSD labels are not trusted.

## Prerequisites

- OpenRouter API key with access to stepfun/step-3.5-flash:free model
- Python 3.10+ with dependencies: `openai`, `python-dotenv`
- NHDSD dataset at project root: `Sarcasm_Headlines_Dataset_v2.json`

```bash
# Set API key
export OPENROUTER_API_KEY="your_key_here"

# Install dependencies (if not already)
uv sync
```

## Execution Steps

### Step 1: Run the Reclassification Script

```bash
uv run python scripts/reclassify_nhdsd_binary.py
```

This will:
1. Load the NHDSD dataset
2. Clean the data (remove duplicates, normalize text)
3. Classify ALL headlines as sarcastic (1) or non-sarcastic (0)
4. Save results incrementally with resume capability

### Step 2: Validate Output

```bash
# Check output file exists and has content
wc -l data/processed/nhdsd_reclassified.jsonl
head -5 data/processed/nhdsd_reclassified.jsonl

# Count label distribution
uv run python -c "
import json
from collections import Counter
counts = Counter()
with open('data/processed/nhdsd_reclassified.jsonl') as f:
    for line in f:
        data = json.loads(line)
        counts[data['is_sarcastic']] += 1
print('Label distribution:')
print(f'  Sarcastic (1): {counts[1]}')
print(f'  Non-sarcastic (0): {counts[0]}')
"
```

## Parallelization Strategy

- **Batch size**: 50 headlines per API request
- **Workers**: 5 parallel threads
- **Rate limit**: 40 requests/minute (1.5s between requests)
- **Estimated throughput**: ~2,000 headlines/hour
- **Total estimated time**: ~14-15 hours for all 28,619 headlines

## Output Files

| File | Description |
|------|-------------|
| `data/processed/nhdsd_cleaned.json` | Cleaned NHDSD dataset (deduplicated) |
| `data/processed/nhdsd_reclassified.jsonl` | Fresh binary labels (JSONL format) |

### Output Schema (JSONL)

```json
{
  "headline": "original headline text",
  "is_sarcastic": 1,
  "confidence": "high|medium|low",
  "model_used": "stepfun/step-3.5-flash:free",
  "article_link": "url",
  "original_label": 0
}
```

## Error Handling & Resume Capability

The script supports automatic resume:

1. **Progress tracking**: Already processed headlines are tracked in output file
2. **On restart**: Script skips previously processed items
3. **Retries**: Failed batches retry up to 5 times with exponential backoff
4. **Rate limit handling**: Automatic detection and backoff for 429 errors

To resume after interruption:
```bash
# Simply re-run the script - it will continue from where it left off
uv run python scripts/reclassify_nhdsd_binary.py
```

## Validation Steps

1. **Count check**: Verify ~28,619 headlines processed
2. **Label distribution**: Check balance between sarcastic/non-sarcastic
3. **Comparison**: Compare with original NHDSD labels to see divergence
4. **Sample verification**: Manually review 20-30 random samples

```bash
# Compare with original labels
uv run python -c "
import json

# Load original
with open('Sarcasm_Headlines_Dataset_v2.json') as f:
    original = {item['headline']: item['is_sarcastic'] for item in json.load(f)}

# Load reclassified
reclassified = {}
with open('data/processed/nhdsd_reclassified.jsonl') as f:
    for line in f:
        item = json.loads(line)
        reclassified[item['headline']] = item['is_sarcastic']

# Calculate agreement
agreement = sum(1 for h in original if original[h] == reclassified.get(h, -1))
print(f'Agreement with original: {agreement}/{len(original)} ({100*agreement/len(original):.1f}%)')
"
```

## Time Estimates

| Phase | Duration | Notes |
|-------|----------|-------|
| Data cleaning | < 1 min | Local processing |
| Classification | ~14-15 hours | API rate limited |
| Validation | 10-15 min | Manual sampling |
| **Total** | **~15 hours** | Mostly automated |

## Monitoring Progress

During execution, the script outputs progress:
```
Loading dataset from Sarcasm_Headlines_Dataset_v2.json...
Total headlines: 28619
Already processed: 0
Remaining to process: 28619

📊 Processing Plan:
   Total batches: 573
   Batch size: 50 headlines
   Rate limit: 40 req/min (1.5s between requests)
   Workers: 5
   Est. time: ~21.5 minutes

Starting processing...

Worker 0: Batch 0 complete (50 items). Total: 50
Worker 1: Batch 1 complete (50 items). Total: 100
...
```

## Troubleshooting

### API Key Issues
```bash
# Verify key is set
echo $OPENROUTER_API_KEY

# Set it if missing
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Rate Limit Exceeded
- Script automatically handles rate limits with backoff
- If consistently hitting limits, reduce `MAX_WORKERS` in script

### Resume Not Working
- Check output file exists: `ls -la data/processed/nhdsd_reclassified.jsonl`
- Verify file has valid JSONL format: `head -1 data/processed/nhdsd_reclassified.jsonl | python -m json.tool`

## Next Steps After Completion

1. Compare new labels with original NHDSD labels
2. Analyze disagreements (which headlines were mislabeled?)
3. Use reclassified data for train/val/test splits
4. Proceed to model fine-tuning (see docs/METHODS.md)

---

_Created: 2026-03-03_
