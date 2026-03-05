# Strategy Augmentation Plan

**Status**: IN PROGRESS  
**Created**: 2026-03-04  
**Model**: stepfun/step-3.5-flash:free (via OpenRouter)

---

> For each generated sarcastic headline in `non_to_sarcastic` pairs, generate the 5 missing strategy variants to create a complete 6-strategy dataset.

## Objective

The current sarcasm pairs dataset contains 14,985 `non_to_sarcastic` pairs where each generated sarcastic headline uses only ONE strategy. This plan augments each headline with the 5 missing strategy variants, creating a dataset where each non-sarcastic source has 6 sarcastic variants (one per strategy type).

**Example**:
- Original: "Company releases software update"
- Existing (satire): "Company proudly releases broken software update"
- Generate 5 more:
  - **sarcasm**: "Great job on the update, everything works perfectly now" (opposite meaning)
  - **irony**: "I love how this update improved my computer's performance" (when it didn't)
  - **understatement**: "The update has some minor issues" (severe understatement)
  - **overstatement**: "This update completely destroyed my computer forever"
  - **rhetorical_question**: "Isn't it wonderful when updates work perfectly on the first try?"

## Current Progress

### Completed Steps

- [x] Augmentation script developed (`scripts/augment_strategy_variants.py`)
- [x] 3 censored headlines filtered out (`data/processed/censored_headlines_strategy.jsonl`)
- [x] Full augmentation run: 74,225 deduplicated variants generated
- [x] Pair splitting: `non_to_sarcastic` (14,948) and `sarcastic_to_non` (13,588) separated
- [x] Merge script: combined original + augmented into complete dataset
- [x] Deduplication fix: removed duplicate strategies per source headline
- [x] Completeness analysis: identified 508 incomplete sources

### Remaining Steps

- [ ] Re-run augmentation for 508 incomplete sources (missing 1-2 strategies each)
- [ ] Re-merge into final complete dataset
- [ ] Create stratified train/val/test splits
- [ ] Update docs/DATASET.md with augmentation details

## Data State (After Augmentation)

**Sources**: 14,948 deduplicated `non_to_sarcastic` headlines (after dedup + censorship removal)

| Metric | Value |
|--------|-------|
| Total sources | 14,948 |
| Complete (6/6 strategies) | 14,440 (96.6%) |
| Incomplete (missing 1-2) | 508 (3.4%) |
| Total augmented variants | 74,225 |
| Merged complete dataset | 89,173 records |

**Missing Strategy Breakdown** (across 508 incomplete sources):
- overstatement: 133 missing
- satire: 120 missing
- rhetorical_question: 99 missing
- sarcasm: 90 missing
- understatement: 84 missing
- irony: 0 missing
## Deliverables

| Artifact | Path | Description |
|----------|------|-------------|
| Augmentation script | `scripts/augment_strategy_variants.py` | Generates 5 variants per headline |
| Split script | `scripts/split_pairs_by_type.py` | Splits pairs by type |
| Merge script | `scripts/merge_augmented_variants.py` | Merges original + augmented |
| Dedup/fix script | `scripts/fix_augmented_variants.py` | Deduplicates augmented strategies |
| Augmented data | `data/processed/sarcasm_pairs_strategy_augmented.jsonl` | 74,225 augmented strategy variants |
| Non-to-sarcastic pairs | `data/processed/sarcasm_pairs_non_to_sarcastic.jsonl` | 14,948 deduplicated source pairs |
| Sarcastic-to-non pairs | `data/processed/sarcasm_pairs_sarcastic_to_non.jsonl` | 13,588 reverse pairs |
| Complete dataset | `data/processed/sarcasm_pairs_non_to_sarcastic_complete.jsonl` | 89,173 merged (needs fix for 508 incomplete) |
| Incomplete sources | `data/processed/incomplete_strategy_sources.jsonl` | 508 sources needing re-augmentation |
| Stats | `data/processed/strategy_augmentation_stats.json` | Distribution and quality metrics |

## Prerequisites

- OpenRouter API key with access to stepfun/step-3.5-flash:free
- Input file: `data/processed/sarcasm_pairs_step35_clean.jsonl`

```bash
# Verify API key
export OPENROUTER_API_KEY="your_key"

# Verify input file exists
wc -l data/processed/sarcasm_pairs_step35_clean.jsonl
# Expected: 28573
```

## 6 Sarcasm Strategies

| Strategy | Definition | Example Transformation |
|----------|------------|------------------------|
| **sarcasm** | Contradicts state, critical tone | Opposite meaning with critical subtext |
| **irony** | Contradicts state, not obviously critical | Says X when clearly not-X, no direct blame |
| **satire** | Appears supportive, contains mockery | Mock-praise that reveals absurdity |
| **understatement** | Undermines importance | Minimizes severity dramatically |
| **overstatement** | Obviously exaggerated | Maximizes severity unrealistically |
| **rhetorical_question** | Question contradicting reality | Question whose implied answer is "obviously no" |

## Processing Logic

### Step 1: Load and Filter

```python
# Load all pairs
pairs = load_jsonl("data/processed/sarcasm_pairs_step35_clean.jsonl")

# Filter to non_to_sarcastic only
target_pairs = [p for p in pairs if p["type"] == "non_to_sarcastic"]
# Result: 14,985 items
```

### Step 2: Determine Missing Strategies

For each pair, identify the existing strategy and which 5 are missing:

```python
ALL_STRATEGIES = ["sarcasm", "irony", "satire", "understatement", "overstatement", "rhetorical_question"]

for pair in target_pairs:
    existing_strategy = pair["strategy"]
    missing_strategies = [s for s in ALL_STRATEGIES if s != existing_strategy]
    # Generate variants for these 5 strategies
```

### Step 3: Batch Generation

Send batches to LLM with specific instructions per strategy:

```python
# Batch size: 5 headlines × 5 strategies = 25 variants per request
# Or more efficiently: 10 headlines × 5 strategies = 50 variants per request

prompt = f"""
Original non-sarcastic headline: "{original}"
Existing sarcastic variant ({existing_strategy}): "{generated}"

Generate 5 NEW sarcastic variants using these strategies:
1. sarcasm: [opposite meaning, critical]
2. irony: [contradiction without direct blame]
3. satire: [mock-praise]
4. understatement: [severe minimization]
5. overstatement: [obvious exaggeration]
6. rhetorical_question: [question implying "obviously no"]

SKIP the existing strategy: {existing_strategy}

Output JSON:
{{
  "variants": [
    {{"strategy": "sarcasm", "headline": "..."}},
    {{"strategy": "irony", "headline": "..."}},
    ... (5 items, excluding {existing_strategy})
  ]
}}
"""
```

### Step 4: Output Format

Each generated variant becomes a new record:

```json
{
  "original_headline": "Company releases software update",
  "non_sarcastic_source": "Company releases software update",
  "generated_headline": "Great job on the update, everything works perfectly now",
  "strategy": "sarcasm",
  "variant_of": "satire",
  "type": "strategy_variant",
  "model_used": "stepfun/step-3.5-flash:free",
  "article_link": "https://..."
}
```

## API Configuration

- **Model**: `stepfun/step-3.5-flash:free` (consistent with original generation)
- **Endpoint**: OpenRouter (`https://openrouter.ai/api/v1`)
- **Temperature**: 0.8 (higher than 0.7 for more creative diversity)
- **Max tokens**: 4000
- **Rate limit**: 40 req/min
- **Batch strategy**: 10 headlines per request × 5 strategies = 50 variants/response

## Parallelization

- **Workers**: 5 threads
- **Batch processing**: 10 source headlines per API call
- **Total API calls**: (14,985 / 10) = ~1,499 batches
- **Rate limit**: 40 req/min = 1.5s between requests
- **Estimated time**: (1,499 × 1.5s) / 60 = ~37 minutes

## Output Schema

### sarcasm_pairs_strategy_augmented.jsonl

```json
{
  "original_headline": "original non-sarcastic headline",
  "non_sarcastic_source": "original non-sarcastic headline (explicit field)",
  "existing_sarcastic": "the previously generated variant",
  "existing_strategy": "satire",
  "generated_headline": "new variant with different strategy",
  "strategy": "sarcasm",
  "type": "strategy_variant",
  "model_used": "stepfun/step-3.5-flash:free",
  "article_link": "url",
  "batch_id": 123
}
```

### strategy_augmentation_stats.json

```json
{
  "metadata": {
    "source_pairs": 14985,
    "variants_generated": 74925,
    "timestamp": "2026-03-04T..."
  },
  "strategy_distribution": {
    "sarcasm": 12500,
    "irony": 12500,
    "satire": 12500,
    "understatement": 12500,
    "overstatement": 12500,
    "rhetorical_question": 12425
  },
  "by_existing_strategy": {
    "satire": { "sarcasm": 2500, "irony": 2500, ... },
    ...
  }
}
```

## Validation

### Count Verification
```bash
# Should have 74,925 new records (14,985 × 5)
wc -l data/processed/sarcasm_pairs_strategy_augmented.jsonl

# Verify strategy distribution is balanced
uv run python -c "
import json
from collections import Counter
counts = Counter()
with open('data/processed/sarcasm_pairs_strategy_augmented.jsonl') as f:
    for line in f:
        item = json.loads(line)
        counts[item['strategy']] += 1
print('Strategy distribution:')
for s, c in counts.most_common():
    print(f'  {s}: {c}')
"
```

### Quality Spot Check
Manually review 20-30 random samples to verify:
1. Each strategy is correctly applied
2. Headlines are genuinely sarcastic
3. Original meaning is preserved/inverted appropriately

## Time Estimates

| Phase | Duration | Notes |
|-------|----------|-------|
| Script development | ~30 min | Adapt from existing scripts |
| Augmentation run | ~37 min | 1,499 API calls at 40 req/min |
| Validation | ~10 min | Manual spot checks |
| **Total** | **~77 min** | ~1.3 hours |

## Troubleshooting

### Resume After Interruption
```bash
# Script tracks progress via output file
# Simply re-run to continue
uv run python scripts/augment_strategy_variants.py
```

### Rate Limit Hit
Reduce `RATE_LIMIT_PER_MINUTE` in script (e.g., to 20) and re-run.

### JSON Parsing Failures
Script includes fallback extraction (markdown blocks, raw find).

## Next Steps

1. **Fix 508 incomplete sources**: Re-run `augment_strategy_variants.py` targeting only the missing strategies from `incomplete_strategy_sources.jsonl`
2. **Re-merge**: Run `merge_augmented_variants.py` to produce final complete dataset
3. **Validate completeness**: Verify all 14,948 sources have exactly 6 strategy variants
4. Create stratified train/val/test splits by strategy
5. Use for multi-strategy style transfer training
6. Update docs/DATASET.md with augmentation details

---

_Created: 2026-03-04_
