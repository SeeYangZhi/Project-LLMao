# Methods

> Approach: strategy classification, fine-tuning, and generation.

## Task Definition

**Input**: A headline (sarcastic OR non-sarcastic)
**Output**: The opposite style version

```
Sarcastic:   "Great job on the update, everything is broken now"
Non-sarcastic: "The update has caused multiple issues"

Non-sarcastic: "The update has caused multiple issues"
Sarcastic:   "Great job on the update, everything is broken now"
```

## Sarcasm Strategies

We use six strategy categories from the **iSarcasm dataset** as control codes:

| Strategy                | Description                                | Example                                         |
| ----------------------- | ------------------------------------------ | ----------------------------------------------- |
| `<sarcasm>`             | Contradicts state of affairs, critical     | "Great job, everything is broken now"           |
| `<irony>`               | Contradicts state of affairs, not critical | "I love waiting in line at the DMV"             |
| `<satire>`              | Appears to support, contains mockery       | "Another meeting that could have been an email" |
| `<understatement>`      | Undermines importance                      | "It's just a minor setback"                     |
| `<overstatement>`       | Obviously exaggerated                      | "I've told you a million times!"                |
| `<rhetorical_question>` | Question contradicting reality             | "Isn't it the best feeling?"                    |

## Approach Overview

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Input     │ ──► │ Strategy Detect │ ──► │  Generate   │
│  "Headline" │     │   <sarcasm>    │     │  Output     │
└─────────────┘     └──────────────────┘     └─────────────┘
```

## Step 1: Data Generation

### 1.1 LLM Generation (Primary)

Use MiniMax M2.5 to generate opposite-style headlines:

- Sarcastic → Non-sarcastic: 13,634 pairs
- Non-sarcastic → Sarcastic: 14,985 pairs
- Total: 28,619 pairs

No manual pairing needed - the LLM generates counterparts directly.

### 1.2 Strategy Annotation

1. **LLM-assisted**: Use few-shot prompting to classify

### 1.3 Control Code Template

```
<sarcasm> Serious headline here
→ Sarcastic version here

<irony> Non-sarcastic headline
→ Sarcastic output
```

## Step 2: Model Fine-Tuning

### 2.1 T5-base (Primary)

**Architecture**: Sequence-to-sequence
**Input**: `<strategy> source_headline`
**Output**: target_headline

```python
# Example
input_text = "<sarcasm> The product has some minor issues"
output_text = "This product is absolutely useless"
```

**Training**:

- Learning rate: 3e-4
- Batch size: 16
- Epochs: 3-5
- Max length: 128 tokens

### 2.2 GPT-2 (Baseline)

**Architecture**: Causal language model
**Input**: `Sarcastic: {source} → Non-sarcastic:`
**Output**: generated text

```python
# Example
prompt = "Sarcastic: Great job on the update → Non-sarcastic:"
# Generates: "The update has caused issues"
```

### 2.3 BART (Baseline)

**Architecture**: Denoising seq2seq
**Training**: Denoise corrupted sarcastic text
**Fine-tune**: On parallel pairs

## Method 3: Inference

### 3.1 Strategy-Aware Generation

```python
def generate(sarcastic_input, strategy=None):
    if strategy is None:
        strategy = detect_strategy(sarcastic_input)

    prompt = f"<{strategy}> {sarcastic_input}"
    output = model.generate(prompt)
    return output
```

### 3.2 Bidirectional Generation

Support both directions:

- Sarcastic → Non-sarcastic
- Non-sarcastic → Sarcastic

## Evaluation Metrics

### Automatic

- **BLEU**: Compare to reference translations
- **Perplexity**: Measure fluency
- **METEOR**: Word overlap with references

### Classifier-Based

- Feed outputs to sarcasm detector
- Check detection rate

### Human Evaluation

- Rate sarcasm effectiveness (1-5)
- Rate naturalness (1-5)
- Identify strategy used

## Implementation Notes

- Log all hyperparameters
- Save checkpoints with config
- Use Weights & Biases for tracking (optional)

## CS4248 Considerations

- Focus on explaining the approach (Why, What, How)
- Ablation studies: Test with/without strategy codes
- Compare multiple models
- Show error analysis

---

_Last updated: 2026-03-02_
