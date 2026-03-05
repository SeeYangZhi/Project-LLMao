# Dataset Audit

> Verified empirical findings from `data/processed/sarcasm_pairs_step35_clean.jsonl`
> Audited: 2026-03-03

---

## 1. Schema and Field Descriptions

**File**: `data/processed/sarcasm_pairs_step35_clean.jsonl`
**Format**: JSON Lines (one JSON object per line)
**Total rows**: 28,333

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `original_headline` | string | The source headline (sarcastic OR non-sarcastic depending on `type`) |
| `generated_headline` | string | The transformed headline (the opposite label) |
| `strategy` | string | Sarcasm type/strategy label (6 values, see below) |
| `type` | string | Direction of transformation: `sarcastic_to_non` or `non_to_sarcastic` |
| `model_used` | string | The LLM that generated the rewrite |
| `article_link` | URL string | Source article URL; used for grouping |

---

## 2. Null / Missing Value Checks

All fields verified for all 28,333 rows:

| Field | Null Count | Notes |
|-------|-----------|-------|
| `original_headline` | 0 | Complete |
| `generated_headline` | 0 | Complete |
| `strategy` | 0 | Complete |
| `type` | 0 | Complete |
| `model_used` | 0 | Complete |
| `article_link` | 0 | Complete |

**No missing values in any field.** Dataset is fully populated.

---

## 3. Allowed Values for `type`

| Value | Count | Description |
|-------|-------|-------------|
| `non_to_sarcastic` | 14,925 | Original is non-sarcastic (HuffPost); generated is sarcastic |
| `sarcastic_to_non` | 13,408 | Original is sarcastic (TheOnion); generated is non-sarcastic |

Both expected values are present. No unexpected or malformed type values found.

---

## 4. Distribution of `strategy`

| Strategy | Count | % of Total |
|----------|-------|-----------|
| sarcasm | 8,699 | 30.7% |
| irony | 6,102 | 21.5% |
| satire | 5,224 | 18.4% |
| overstatement | 3,976 | 14.0% |
| understatement | 3,295 | 11.6% |
| rhetorical_question | 1,037 | 3.7% |
| **Total** | **28,333** | **100%** |

**Class imbalance**: `rhetorical_question` is 8.4x less frequent than `sarcasm`. This is a critical imbalance for the type classification task. Macro-F1 as primary metric is the correct choice; class-weighted loss will be needed in BERT training.

---

## 5. Model Used Distribution

| Model | Count |
|-------|-------|
| `stepfun/step-3.5-flash:free` | 28,333 |

**Only one model** was used for all rewrites. No multi-model bias; this simplifies analysis.

---

## 6. Duplicate Checks

### Duplicate Original Headlines
- **Count**: 70 duplicate original_headlines (across 28,333 rows)
- **Cause**: Same TheOnion or HuffPost headline appearing under different `article_link` values (syndicated content) or slight variants
- **Action**: These are different JSONL rows (different article_links), so they get different `pair_id`. No action needed; grouping by `article_link` will handle same-article pairs.

### Duplicate Generated Headlines
- **Count**: 1 duplicate generated_headline
- **Cause**: LLM produced identical rewrite for two different inputs
- **Action**: Both rows retained; they have different original headlines and are distinguishable by pair_id

### Duplicate Article Links
- **Count**: 2 duplicate article_links (i.e., 2 article_link values appear more than once)
- **Cause**: Same article processed twice with both `sarcastic_to_non` and `non_to_sarcastic` transformations, OR same article URL with different strategy annotations
- **Action**: These duplicates define the `group_id`; rows sharing an article_link will be grouped together and placed in the same split, preventing leakage.

---

## 7. Label Derivation Logic

### Rule (Non-Negotiable)

For each JSONL row `r`:

```
if r['type'] == 'sarcastic_to_non':
    sample_A: text = r['original_headline'],  binary_label = 1 (sarcastic),  type_label = r['strategy']
    sample_B: text = r['generated_headline'], binary_label = 0 (non-sarcastic), type_label = None

if r['type'] == 'non_to_sarcastic':
    sample_A: text = r['original_headline'],  binary_label = 0 (non-sarcastic), type_label = None
    sample_B: text = r['generated_headline'], binary_label = 1 (sarcastic),     type_label = r['strategy']
```

### Derived Binary Dataset Size

| Class | From `sarcastic_to_non` | From `non_to_sarcastic` | Total |
|-------|------------------------|------------------------|-------|
| Sarcastic (1) | 13,408 (original) | 14,925 (generated) | 28,333 |
| Non-sarcastic (0) | 13,408 (generated) | 14,925 (original) | 28,333 |
| **Total** | | | **56,666** |

**Binary dataset is perfectly balanced**: exactly 28,333 sarcastic and 28,333 non-sarcastic samples.

### Derived Type Dataset Size

Only sarcastic samples (binary_label == 1) are included in the type dataset:

| Strategy | Count |
|----------|-------|
| sarcasm | 8,699 |
| irony | 6,102 |
| satire | 5,224 |
| overstatement | 3,976 |
| understatement | 3,295 |
| rhetorical_question | 1,037 |
| **Total** | **28,333** |

---

## 8. Leakage Risk Analysis

### Risk Description

Each JSONL row produces a **semantically coupled pair**: an original headline and its rewrite share the same topic, named entities, and much vocabulary. If a random split places the sarcastic version of a pair in train and the non-sarcastic version in test:
- The model sees the topic/entities in training
- The test sample is not truly "unseen" — it shares semantic content with a training sample
- Performance is inflated; generalization is not measured

**This is guaranteed to occur with naive (random) splitting after expansion.**

### Leakage Risk Level by Source

| Risk Source | Severity | Mitigation |
|-------------|----------|-----------|
| Same pair across splits | **Critical** | Pair-level split (pair never crosses split boundary) |
| Same article multiple directions | **High** | Group by article_link; all rows from same URL in one split |
| Similar topics (different articles) | **Low** | Cannot perfectly control; acceptable |

### Final Split Grouping Decision

**Primary grouping key**: `article_link` (normalized: lowercase, strip trailing slash)

Rationale:
- All rows sharing the same article URL are semantically coupled
- 2 duplicate article_links found → these rows must cohabit the same split
- ~28,333 unique group_ids (near-unique since only 2 duplicates)

**Fallback**: `pair_id` = row index (if article_link is missing or malformed)

Since no null article_links exist and only 2 duplicates, the primary grouping is reliable.

---

## 9. Data Quality Issues / Anomalies

### Generation Artifacts

Generated headlines may contain:
- More formal/neutral phrasing than natural non-sarcastic writing
- Slight topic shifts (LLM reframing rather than pure rewrite)
- Occasional metadata bleeding (e.g., "According to reports..." framing artifacts)

**Recommendation**: Track `is_generated` flag in the binary dataset to allow artifact-aware error analysis.

### Strategy Label Granularity

The `rhetorical_question` strategy label is the most syntactically distinct (ends with `?`). The other 5 strategies (`sarcasm`, `irony`, `satire`, `overstatement`, `understatement`) may have substantial semantic overlap, making multiclass type classification challenging.

The label `sarcasm` as a strategy within a sarcasm dataset is potentially circular/ambiguous. This should be flagged in error analysis.

### Single Model Bias

All generated headlines come from `stepfun/step-3.5-flash:free`. The classifier may learn to distinguish "Step-3.5 generation style" from "authentic headline style" rather than true sarcasm. Error analysis should check for this artifact.

---

## 10. Verification Checklist

- [x] All 28,333 rows parsed without error
- [x] No null values in any field
- [x] Both `type` values present and expected
- [x] All 6 strategy values identified
- [x] Duplicate checks completed (70 orig_headline dupes, 1 gen_headline dupe, 2 article_link dupes)
- [x] Label derivation rules documented and validated
- [x] Binary dataset size computed: 56,666 samples, perfectly balanced
- [x] Type dataset size computed: 28,333 samples, imbalanced (3.7% to 30.7%)
- [x] Leakage risk identified and mitigation strategy defined
- [x] Group ID strategy confirmed: `article_link` primary, `pair_id` fallback
