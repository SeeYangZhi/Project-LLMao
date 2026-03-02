# Experiments

> Experiment tracking and results for Project LLMao.

## Experiment Tracking

Use a consistent format for all experiments:

```yaml
experiment_id: "exp_001"
date: "2026-03-02"
model: "t5-base"
dataset: "nhdsd_paired_v1"
strategy: "all"  # or specific: sarcasm, irony, satire, understatement, overstatement, rhetorical_question

hyperparameters:
  learning_rate: 3e-4
  batch_size: 16
  epochs: 3
  max_length: 128
  seed: 42

results:
  bleu_score: 0.XX
  perplexity: XX.XX
  classifier_detection_rate: 0.XX
```

## Baseline Experiments

### Exp 1: T5-base (No Control Codes)

- **Hypothesis**: Basic seq2seq without strategy info
- **Setup**: `source → target` direct mapping
- **Expected**: Lower interpretability

### Exp 2: T5-base (With Control Codes)

- **Hypothesis**: Strategy tokens improve quality
- **Setup**: `<strategy> source → target`
- **Compare**: vs Exp 1

### Exp 3: GPT-2 Baseline

- **Hypothesis**: Causal LM can work for style transfer
- **Setup**: `Sarcastic: {x} → Non-sarcastic: {y}`
- **Compare**: vs T5

### Exp 4: BART Baseline

- **Hypothesis**: Denoising approach for style transfer
- **Setup**: Fine-tune on parallel pairs
- **Compare**: vs T5, GPT-2

## Ablation Studies

### Ablation 1: Strategy Categories

| Variant | Description |
|---------|-------------|
| All strategies | Use all 6 control codes (sarcasm, irony, satire, understatement, overstatement, rhetorical_question) |
| No strategy | No control codes |
| Single strategy | Only sarcasm |

### Ablation 2: Data Size

| Variant | Training Size |
|---------|---------------|
| Full | ~28K pairs |
| Half | ~14K pairs |
| Quarter | ~7K pairs |

### Ablation 3: Augmentation

| Variant | Description |
|---------|-------------|
| No augmentation | Raw NHDSD only |
| LLM augmented | + LLM-generated pairs |

## Results Template

### Table 1: Model Comparison

| Model | BLEU ↑ | Perplexity ↓ | Detection Rate ↑ |
|-------|--------|--------------|-------------------|
| T5-base | 0.XX | XX.XX | 0.XX |
| GPT-2 | 0.XX | XX.XX | 0.XX |
| BART | 0.XX | XX.XX | 0.XX |

### Table 2: Ablation - Control Codes

| Variant | BLEU | Human Score |
|---------|------|-------------|
| With codes | 0.XX | X.X/5 |
| Without codes | 0.XX | X.X/5 |

### Table 3: Strategy Breakdown

| Strategy | BLEU | Detection Rate |
|----------|------|----------------|
| sarcasm | 0.XX | 0.XX |
| irony | 0.XX | 0.XX |
| satire | 0.XX | 0.XX |
| understatement | 0.XX | 0.XX |
| overstatement | 0.XX | 0.XX |
| rhetorical_question | 0.XX | 0.XX |

## Error Analysis

Categories of failures:

1. **Strategy misclassification**
   - Model generates wrong strategy
   - Impact: Lower quality output

2. **Fluency issues**
   - Grammar errors
   - Impact: Low perplexity but unnatural

3. **Over-generation**
   - Too exaggerated or too subtle
   - Impact: Detection rate changes

4. **Semantic drift**
   - Meaning changes during transfer
   - Impact: Truthfulness issues

## Experiment Log

| Date | Experiment | Model | Key Finding |
|------|------------|-------|--------------|
| YYYY-MM-DD | Baseline | T5-base | Initial results |
| YYYY-MM-DD | +Control Codes | T5-base | Improved interpretability |

---

*Last updated: 2026-03-02*
