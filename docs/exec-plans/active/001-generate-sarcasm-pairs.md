# Execution Plan: Generate Sarcasm Pairs with MiniMax M2.5

**Plan ID**: exec-001
**Status**: ACTIVE
**Created**: 2026-03-02
**Model**: MiniMax M2.5 Free (opencode/minimax-m2.5-free)

## Objective

Generate parallel sarcastic/non-sarcastic headline pairs from the NHDSD dataset using MiniMax M2.5 via OpenCode Zen API.

## Rationale

The NHDSD dataset contains:
- 13,634 sarcastic headlines from TheOnion
- 14,985 non-sarcastic headlines from HuffPost
- **Total: 28,619 headlines**

These are NOT naturally paired - they're from different articles. To train a style transfer model, we need parallel pairs. MiniMax M2.5 Free offers:
- **Free** (no API cost)
- **200K context** (handle long prompts)
- **Fast inference** (~100 tokens/sec)
- **Strong coding/reasoning** (80.2% SWE-bench)

## Approach

Use few-shot prompting with MiniMax to generate non-sarcastic versions of sarcastic headlines (and vice versa).

### Strategy Categories (from iSarcasm Dataset)

Based on the iSarcasmEval dataset which has complete category annotations:

| Tag | Description | Example |
|-----|-------------|--------|
| `<sarcasm>` | Contradicts state of affairs, critical towards addressee | "Great job on the update, everything is broken now" |
| `<irony>` | Contradicts state of affairs, not obviously critical | "I love waiting in line at the DMV for hours" |
| `<satire>` | Appears to support, but contains mockery | "Wow, another meeting that could have been an email" |
| `<understatement>` | Undermines importance of situation | "It's just a minor setback" (for a disaster) |
| `<overstatement>` | Obviously exaggerated terms | "I've told you a million times!" |
| `<rhetorical_question>` | Question with inference contradicting reality | "Isn't it just the best feeling to be ignored?" |

### Prompt Template

```
You are a sarcasm style transfer expert. Given a headline, generate the opposite style version.

Sarcasm Strategies (from iSarcasm dataset):
- sarcasm: Contradicts state of affairs, critical towards addressee
- irony: Contradicts state of affairs, not obviously critical
- satire: Appears to support, but contains mockery
- understatement: Undermines importance
- overstatement: Obviously exaggerated terms
- rhetorical_question: Question with inference contradicting reality

Example 1:
Sarcastic: "Great job on the update, everything is broken now"
Non-sarcastic: "The latest update has caused multiple issues"
Strategy: sarcasm

Example 2:
Sarcastic: "I love waiting in line at the DMV for hours"
Non-sarcastic: "Waiting at the DMV takes several hours"
Strategy: irony

Now generate for:
Sarcastic: "{headline}"
Non-sarcastic:
```

## Technical Details

### API Configuration
- **Endpoint**: `https://opencode.ai/zen/v1/chat/completions`
- **Model**: `minimax-m2.5-free`
- **Format**: OpenAI-compatible (`@ai-sdk/openai-compatible`)

### Implementation Steps

1. **Set up API client**
   - Get OpenCode Zen API key
   - Configure Python client with openai-compatible SDK

2. **Load dataset**
   - Read NHDSD JSON
   - Filter sarcastic headlines (is_sarcastic=1)

3. **Batch generation**
   - Process in batches (avoid rate limits)
   - Use few-shot prompting
   - Track strategy type

4. **Save outputs**
   - Store as JSONL: `{"sarcastic": "...", "non_sarcastic": "...", "strategy": "..."}`
   - Log API costs (should be free)

## Expected Outputs

- 13,634 sarcastic → non-sarcastic pairs
- 14,985 non-sarcastic → sarcastic pairs
- **Total: 28,619 pairs**
- Strategy annotation for each pair
- File: `data/processed/sarcasm_pairs_minimax.jsonl`

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| API rate limits | Batch processing with delays |
| Quality issues | Sample human evaluation |
| Model unavailable | Fall back to local LLM |

## Timeline

- [ ] Step 1: API setup (5 min)
- [ ] Step 2: Load & filter data (10 min)  
- [ ] Step 3: Generate pairs (1-2 hrs)
- [ ] Step 4: Quality check (15 min)
- [ ] Step 5: Save & document (10 min)

---

*Last updated: 2026-03-02*
