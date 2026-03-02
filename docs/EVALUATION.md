# Evaluation (TO BE UPDATED)

> Metrics, evaluation protocols for Project LLMao.

## Automatic Metrics

### BLEU Score

```python
from nltk.translate.bleu_score import sentence_bleu

reference = ["The update has caused multiple issues"]
candidate = model.generate(input_text)

score = sentence_bleu(reference, candidate)
```

**Target**: Higher is better
**Typical range**: 0.1 - 0.5 for generation tasks

### Perplexity

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokens = tokenizer.encode(text)
loss = model(tokens, labels=tokens).loss
perplexity = math.exp(loss)
```

**Target**: Lower is better
**Typical range**: 10 - 100 for news-style text

### METEOR

```python
from nltk.translate.meteor_score import meteor_score

score = meteor_score(reference, candidate)
```

**Target**: Higher is better
**Note**: More flexible than BLEU

### Classifier-Based Evaluation

Feed generated outputs to a sarcasm classifier:

```python
detector = load_sarcasm_detector()
original_sarcastic = detector.predict(original_sarcastic_text)
generated_sarcastic = detector.predict(generated_text)

detection_rate = sum(generated_sarcastic) / len(generated_sarcastic)
```

**Hypothesis**: If generation worked, outputs should be detected at similar rate to originals

## Evaluation Metrics Summary

| Metric         | What It Measures  | Target | Required? |
| -------------- | ----------------- | ------ | --------- |
| BLEU           | Word overlap      | Higher | Yes       |
| Perplexity     | Fluency           | Lower  | Yes       |
| METEOR         | Flexible overlap  | Higher | Optional  |
| Detection Rate | Sarcasm preserved | Higher | Yes       |

## CS4248 Report Requirements

For the final report:

1. **Present all metrics**: Both automated and human
2. **Show error analysis**: Categorize failures
3. **Discuss limitations**: What can't current metrics capture?
4. **Ablation studies**: Test component contributions

### Example Results Table

```
                    BLEU    PPL    Detection
T5-base (ours)     0.23    45.2   0.78
GPT-2 (baseline)   0.19    52.1   0.71
BART (baseline)    0.21    48.3   0.74
```

## Reproducibility in Evaluation

- Use fixed random seed for sampling
- Report number of samples evaluated
- Include sample outputs in appendix

---

_Last updated: 2026-03-02_
