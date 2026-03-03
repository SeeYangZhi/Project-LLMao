# Execution Plan: Generate Sarcasm Pairs

**Status**: COMPLETED
**Created**: 2026-03-02
**Completed**: 2026-03-03
**Model**: StepFun Step 3.5 Flash Free (via OpenRouter)

## Results

- Sarcastic → non-sarcastic pairs
- Non-sarcastic → sarcastic pairs
- Strategy annotations included for all generated pairs

### Unprocessed Headlines (8)

6 headlines are permanently blocked by StepFun's **input-level censorship filter** (HTTP 451). The filter rejects the request before the model sees it — prompt engineering cannot bypass this. These contain explicit/sexual content from TheOnion satirical articles.

Stored in: `data/processed/remaining_headlines.jsonl`

## Output Files

| File                                              | Description                 | Size |
| ------------------------------------------------- | --------------------------- | ---- |
| `data/processed/sarcasm_pairs_step35_clean.jsonl` | Clean output (no reasoning) | 11MB |
| `data/processed/remaining_headlines.jsonl`        | 8 unprocessable headlines   | <1KB |

## Objective

Generate parallel sarcastic/non-sarcastic headline pairs from the NHDSD dataset using free LLM APIs.

## Rationale

The NHDSD dataset contains:

- 13,634 sarcastic headlines from TheOnion
- 14,985 non-sarcastic headlines from HuffPost
- **Total: 28,619 headlines**

These are NOT naturally paired — they're from different articles. To train a style transfer model, we need parallel pairs.

## Approach

Used few-shot prompting to generate non-sarcastic versions of sarcastic headlines (and vice versa). Batched headlines in groups for efficiency.

### Strategy Categories (from iSarcasm Dataset)

| Tag                   | Description                                              |
| --------------------- | -------------------------------------------------------- |
| `sarcasm`             | Contradicts state of affairs, critical towards addressee |
| `irony`               | Contradicts state of affairs, not obviously critical     |
| `satire`              | Appears to support, but contains mockery                 |
| `understatement`      | Undermines importance of situation                       |
| `overstatement`       | Obviously exaggerated terms                              |
| `rhetorical_question` | Question with inference contradicting reality            |

## Technical Details

### API Configuration

- **Endpoint**: OpenRouter (`https://openrouter.ai/api/v1`)
- **Model**: `stepfun/step-3.5-flash:free`
- **Format**: OpenAI-compatible

### Scripts

- `scripts/generate_sarcasm_pairs_parallel.py` — Main parallel batch processing script

## Risks Encountered

| Risk              | Outcome                                   |
| ----------------- | ----------------------------------------- |
| API rate limits   | Handled with batching + delays            |
| Content filtering | 8 headlines blocked by input-level filter |

---

_Last updated: 2026-03-03_
