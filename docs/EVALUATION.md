# Evaluation Pipeline Documentation

## Project LLMao — CS4248 Team 14

**Task:** Sarcasm Style Transfer — convert sarcastic headlines to non-sarcastic while preserving meaning

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The 7-Metric Pipeline](#the-7-metric-pipeline)
3. [Full Dataset Results (2857 Samples × 14 Models)](#full-dataset-results)
4. [Human Evaluation (140 Samples × 3 Models)](#human-evaluation)
5. [Multi-Classifier Comparison Study](#multi-classifier-comparison-study)
6. [Subtype Analysis](#subtype-analysis)
7. [Deep Analysis: Why These Results Occur](#deep-analysis-why-these-results-occur)
8. [Key Findings](#key-findings)
9. [Conclusions](#conclusions)

---

## Executive Summary

### Evaluation Scale

- **Full dataset:** 2857 samples × 14 models × 3 classifiers × 7 metrics
- **Human evaluation:** 140 samples × 3 models × 2 annotators

### Core Findings

| Finding | Evidence |
|---------|----------|
| All classifiers unreliable | κ = -0.08 to +0.10 vs human (κ > 0.8) |
| T5-Joint best meaning preservation | 16.4% meaning change vs 25% Control, 41% BART-RL |
| LLaMA destroys meaning | Similarity 0.66, LLM meaning 3.34/5 |
| Ablation models perform similarly | Minimal difference when removing subtypes |
| Different subtypes have different failure modes | Satire hardest (80% miss rate), generic sarcasm easiest (44% miss rate), overstatement is MODEL failure |

---

## The 7-Metric Pipeline

Our evaluation pipeline (`scripts/eval_pipeline.py`) computes 7 complementary metrics:

### Metric 1: Sarcasm Flip Rate

**What it measures:** Does the classifier think sarcasm was removed?

**Three classifiers tested:**

| Classifier | Architecture | Training Data | Model ID |
|------------|--------------|---------------|----------|
| RoBERTa-Twitter | RoBERTa | Twitter irony | [`cardiffnlp/twitter-roberta-base-irony`](https://huggingface.co/cardiffnlp/twitter-roberta-base-irony) |
| Bert-Kaggle | BERT | Kaggle headlines | [`helinivan/english-sarcasm-detector`](https://huggingface.co/helinivan/english-sarcasm-detector) |
| RoBERTa-News | RoBERTa | News headlines | [`jkhan447/sarcasm-detection-RoBerta-base-POS`](https://huggingface.co/jkhan447/sarcasm-detection-RoBerta-base-POS) |

**Interpretation:** Higher flip rate = more outputs classified as non-sarcastic. BUT our study shows classifiers are unreliable (see Section 5).

### Metric 2: Semantic Similarity

**Tool:** `sentence-transformers/all-MiniLM-L6-v2`

**What it measures:** Is the core meaning preserved between input and output?

| Score | Interpretation |
|-------|----------------|
| 0.95+ | Nearly identical (possibly just paraphrased) |
| 0.85–0.95 | Good meaning preservation ✓ |
| 0.70–0.85 | Moderate drift |
| < 0.70 | Significant meaning loss ✗ |

### Metric 3: Perplexity

**Tool:** GPT-2

**What it measures:** Is the output fluent, natural English?

| Score | Interpretation |
|-------|----------------|
| < 300 | Very fluent |
| 300–600 | Normal |
| 600–1000 | Somewhat disfluent |
| > 1000 | Problematic |

### Metric 4: BLEU vs Input

**Tool:** `sacrebleu`

**What it measures:** N-gram overlap between output and INPUT (not reference)

**Purpose:** Detects **copying** vs genuine rewriting

| BLEU | Similarity | Interpretation |
|------|------------|----------------|
| High (> 0.20) | High | Paraphrasing (minimal change) |
| Low (< 0.05) | High | Genuine rewriting ✓ |
| Low | Low | Meaning lost ✗ |

### Metric 5: Edit Distance

**Tool:** Word-level Levenshtein distance, normalized to [0, 1]

**What it measures:** How much was the text modified?

| Score | Interpretation |
|-------|----------------|
| 0.0–0.3 | Minor edits (punctuation, casing) |
| 0.4–0.6 | Moderate rewriting |
| 0.7–0.9 | Significant rewriting |
| 0.9+ | Complete rewrite |

### Metric 6: LLM-as-Judge

**Tool:** Gemini 2.5 Flash (batch of 50 samples per model)

**Three dimensions scored 1–5:**

| Dimension | Question |
|-----------|----------|
| sarcasm_removed | Is the output non-sarcastic? |
| meaning_preserved | Is the core meaning intact? |
| fluency | Is it natural English? |

### Metric 7: Paraphrase Score

**Formula:** `similarity × (1 - BLEU_vs_input)`

**What it measures:** Genuine rewriting that preserves meaning

| Score | Interpretation |
|-------|----------------|
| > 0.20 | Good: high similarity + low copying |
| 0.10–0.20 | Moderate |
| < 0.05 | Poor: either copying or meaning lost |

**Why we invented this metric:**

Existing metrics fail individually:

| Scenario | Similarity | BLEU | Problem |
|----------|------------|------|---------|
| Just lowercase input | 0.99 | 0.95 | Looks good but no real change |
| Complete rewrite | 0.65 | 0.02 | Looks bad but might be necessary |

**Concrete example:**

- Input: "Man Shocked By Obvious Fact"
- Output A: "man shocked by obvious fact" → Similarity 0.99, BLEU 0.95 → **Para: 0.05 (BAD — just copied)**
- Output B: "A person was surprised to learn something widely known" → Similarity 0.85, BLEU 0.08 → **Para: 0.78 (GOOD — genuine rewrite)**

The paraphrase score captures what neither metric alone can detect: genuine rewriting (low BLEU) that still preserves meaning (high similarity).

---

## Full Dataset Results

### Evaluation Scale

- **2857 samples** per model
- **14 models** evaluated
- **3 classifiers** × **7 metrics** per model

### Complete 7-Metric Summary (All 14 Models)

| Model | Flip Rate* | Similarity | Perplexity | BLEU | Edit Dist | Para Score | LLM Sarc | LLM Mean | LLM Flu |
|-------|------------|------------|------------|------|-----------|------------|----------|----------|---------|
| **t5_base_joint** | 8.4% | **0.870** | 571.2 | 0.188 | 0.630 | 0.170 | 3.70 | **4.82** | 4.48 |
| **t5_control** | 8.1% | **0.878** | 613.7 | 0.216 | 0.592 | 0.197 | 3.70 | 4.64 | 4.28 |
| joint | 8.4% | **0.872** | 634.9 | 0.211 | 0.600 | 0.192 | 3.82 | 4.04 | **4.74** |
| **bart_base_rl** | 8.8% | 0.852 | 726.3 | 0.216 | 0.606 | 0.198 | 4.10 | 4.26 | 4.38 |
| bart_base | 8.4% | 0.853 | **517.9** | 0.160 | 0.661 | 0.143 | 3.58 | 4.48 | 4.44 |
| bart_base_ce | 13.1% | 0.636 | 364.4 | 0.023 | 0.923 | 0.018 | 4.84 | 3.52 | 4.52 |
| bart_base_ce_rl | 12.6% | 0.609 | 457.4 | 0.021 | 0.928 | 0.015 | 4.56 | 2.90 | 4.36 |
| llama_3_2_1b | 13.7% | 0.656 | 377.8 | 0.013 | **0.948** | 0.009 | **5.00** | 3.34 | **4.92** |
| abl_without_irony | 8.0% | **0.882** | 591.4 | **0.234** | 0.575 | **0.213** | 3.86 | 4.60 | 4.26 |
| abl_without_overstatement | 8.4% | **0.885** | 606.7 | **0.235** | **0.570** | **0.215** | 3.28 | 4.58 | 4.48 |
| abl_without_rhet_q | 8.2% | **0.881** | 599.3 | **0.234** | 0.573 | **0.214** | 3.18 | 4.42 | **4.86** |
| abl_without_sarcasm | 7.7% | **0.883** | 607.8 | **0.237** | **0.568** | **0.217** | 3.32 | 4.24 | 4.14 |
| abl_without_satire | 8.0% | **0.880** | 593.5 | 0.229 | 0.579 | 0.209 | 3.18 | 4.52 | **4.86** |
| abl_without_understatement | 8.2% | **0.881** | **589.9** | 0.226 | 0.580 | 0.207 | 3.52 | 4.44 | 4.48 |

*Flip Rate shown for RoBERTa-Twitter. Bold = notable values.

### Model Rankings by Each Metric

#### Best Semantic Similarity (Meaning Preservation)

| Rank | Model | Similarity |
|------|-------|------------|
| 1 | abl_without_overstatement | 0.885 |
| 2 | abl_without_sarcasm | 0.883 |
| 3 | abl_without_irony | 0.882 |
| 4 | abl_without_rhet_q | 0.881 |
| 5 | abl_without_understatement | 0.881 |
| 6 | abl_without_satire | 0.880 |
| 7 | t5_control | 0.878 |
| 8 | joint | 0.872 |
| 9 | t5_base_joint | 0.870 |
| 10 | bart_base | 0.853 |
| 11 | bart_base_rl | 0.852 |
| 12 | llama_3_2_1b | 0.656 |
| 13 | bart_base_ce | 0.636 |
| 14 | bart_base_ce_rl | 0.609 |

#### Best LLM Meaning Preserved Score

| Rank | Model | LLM Meaning |
|------|-------|-------------|
| 1 | t5_base_joint | 4.82 |
| 2 | t5_control | 4.64 |
| 3 | abl_without_irony | 4.60 |
| 4 | abl_without_overstatement | 4.58 |
| 5 | abl_without_satire | 4.52 |
| 6 | bart_base | 4.48 |
| 7 | abl_without_understatement | 4.44 |
| 8 | abl_without_rhet_q | 4.42 |
| 9 | bart_base_rl | 4.26 |
| 10 | abl_without_sarcasm | 4.24 |
| 11 | joint | 4.04 |
| 12 | bart_base_ce | 3.52 |
| 13 | llama_3_2_1b | 3.34 |
| 14 | bart_base_ce_rl | 2.90 |

#### Best LLM Sarcasm Removed Score

| Rank | Model | LLM Sarcasm |
|------|-------|-------------|
| 1 | llama_3_2_1b | 5.00 |
| 2 | bart_base_ce | 4.84 |
| 3 | bart_base_ce_rl | 4.56 |
| 4 | bart_base_rl | 4.10 |
| 5 | abl_without_irony | 3.86 |
| 6 | joint | 3.82 |
| 7 | t5_base_joint | 3.70 |
| 8 | t5_control | 3.70 |
| 9 | bart_base | 3.58 |
| 10 | abl_without_understatement | 3.52 |
| 11 | abl_without_sarcasm | 3.32 |
| 12 | abl_without_overstatement | 3.28 |
| 13 | abl_without_rhet_q | 3.18 |
| 14 | abl_without_satire | 3.18 |

#### Best LLM Fluency Score

| Rank | Model | LLM Fluency |
|------|-------|-------------|
| 1 | llama_3_2_1b | 4.92 |
| 2 | abl_without_rhet_q | 4.86 |
| 3 | abl_without_satire | 4.86 |
| 4 | joint | 4.74 |
| 5 | bart_base_ce | 4.52 |
| 6 | t5_base_joint | 4.48 |
| 7 | abl_without_overstatement | 4.48 |
| 8 | abl_without_understatement | 4.48 |
| 9 | bart_base | 4.44 |
| 10 | bart_base_rl | 4.38 |
| 11 | bart_base_ce_rl | 4.36 |
| 12 | t5_control | 4.28 |
| 13 | abl_without_irony | 4.26 |
| 14 | abl_without_sarcasm | 4.14 |

#### Lowest Perplexity (Most Fluent)

| Rank | Model | Perplexity |
|------|-------|------------|
| 1 | bart_base_ce | 364.4 |
| 2 | llama_3_2_1b | 377.8 |
| 3 | bart_base_ce_rl | 457.4 |
| 4 | bart_base | 517.9 |
| 5 | t5_base_joint | 571.2 |
| 6 | abl_without_understatement | 589.9 |
| 7 | abl_without_irony | 591.4 |
| 8 | abl_without_satire | 593.5 |
| 9 | abl_without_rhet_q | 599.3 |
| 10 | abl_without_overstatement | 606.7 |
| 11 | abl_without_sarcasm | 607.8 |
| 12 | t5_control | 613.7 |
| 13 | joint | 634.9 |
| 14 | bart_base_rl | 726.3 |

#### Best Paraphrase Score (Genuine Rewriting)

| Rank | Model | Para Score |
|------|-------|------------|
| 1 | abl_without_sarcasm | 0.217 |
| 2 | abl_without_overstatement | 0.215 |
| 3 | abl_without_rhet_q | 0.214 |
| 4 | abl_without_irony | 0.213 |
| 5 | abl_without_satire | 0.209 |
| 6 | abl_without_understatement | 0.207 |
| 7 | bart_base_rl | 0.198 |
| 8 | t5_control | 0.197 |
| 9 | joint | 0.192 |
| 10 | t5_base_joint | 0.170 |
| 11 | bart_base | 0.143 |
| 12 | bart_base_ce | 0.018 |
| 13 | bart_base_ce_rl | 0.015 |
| 14 | llama_3_2_1b | 0.009 |

### Multi-Classifier Flip Rates (All 3 Classifiers)

| Model | RoBERTa-Twitter | Bert-Kaggle | RoBERTa-News | Spread |
|-------|-----------------|-------------------|--------------|--------|
| t5_base_joint | 8.4% | 41.6% | 21.2% | 33.2 pp |
| t5_control | 8.1% | 40.9% | 20.3% | 32.9 pp |
| joint | 8.4% | 41.1% | 22.4% | 32.7 pp |
| bart_base_rl | 8.8% | 39.9% | 26.4% | 31.1 pp |
| bart_base | 8.4% | 41.8% | 21.7% | 33.4 pp |
| bart_base_ce | 13.1% | 17.9% | 3.4% | 14.5 pp |
| bart_base_ce_rl | 12.6% | 39.6% | 8.2% | 31.4 pp |
| llama_3_2_1b | 13.7% | 16.3% | 4.8% | 11.5 pp |
| abl_without_irony | 8.0% | 40.7% | 18.8% | 32.7 pp |
| abl_without_overstatement | 8.4% | 40.8% | 18.9% | 32.4 pp |
| abl_without_rhet_q | 8.2% | 40.6% | 19.0% | 32.4 pp |
| abl_without_sarcasm | 7.7% | 41.2% | 19.3% | 33.4 pp |
| abl_without_satire | 8.0% | 41.2% | 19.2% | 33.2 pp |
| abl_without_understatement | 8.2% | 41.0% | 18.9% | 32.8 pp |
| **AVERAGE** | **9.3%** | **37.5%** | **17.3%** | **28.2 pp** |

**Critical:** Same model, different classifier → up to 33 percentage point spread!

### Model Archetypes

Based on all 7 metrics, models cluster into distinct behavioral patterns:

**Type A: Conservative Rewriters (T5, Ablations)**
- High similarity (0.87–0.89)
- Moderate edit distance (0.57–0.63)
- High BLEU (0.19–0.24)
- Good meaning preservation (LLM 4.2–4.8)
- Moderate sarcasm removal (LLM 3.2–3.9)
- Good paraphrase score (0.17–0.22)

**Type B: Aggressive Rewriters (BART-CE, LLaMA)**
- Low similarity (0.61–0.66)
- High edit distance (0.92–0.95)
- Very low BLEU (0.01–0.02)
- Poor meaning preservation (LLM 2.9–3.5)
- High sarcasm removal (LLM 4.6–5.0)
- Very low paraphrase score (0.01–0.02)

**Type C: Balanced (BART-base, BART-RL)**
- Moderate similarity (0.85–0.86)
- Moderate edit distance (0.60–0.66)
- Moderate BLEU (0.16–0.22)
- Good meaning preservation (LLM 4.3–4.5)
- Moderate sarcasm removal (LLM 3.6–4.1)
- Moderate paraphrase score (0.14–0.20)

---

## Human Evaluation

### Protocol

- **140 samples** per model (stratified by sarcasm subtype)
- **3 models:** T5-Joint, T5-Control, BART-RL
- **2 independent annotators** per model
- **Labels:** `sarcasm_removed` (Y/N), `meaning_change` (Y/N)

### Inter-Annotator Agreement

| Model | Raw Agreement | Cohen's κ | Interpretation |
|-------|---------------|-----------|----------------|
| T5-Joint | 92.1% | 0.839 | Excellent |
| T5-Control | 94.3% | 0.883 | Excellent |
| BART-RL | 94.3% | 0.884 | Excellent |

**All κ > 0.8 = Human labels are reliable ground truth**

### Human Evaluation Results

| Metric | T5-Joint | T5-Control | BART-RL |
|--------|----------|------------|---------|
| Human Flip Rate | 54.3% | 54.3% | 52.9% |
| Meaning Change | **16.4%** | 25.0% | **40.7%** |
| Strict Success | **43.6%** | 39.3% | 34.3% |

**Strict Success** = Sarcasm removed AND meaning preserved

### Human vs Automated Metrics Comparison

| Metric | T5-Joint | T5-Control | BART-RL |
|--------|----------|------------|---------|
| **Human Flip Rate** | **54.3%** | **54.3%** | **52.9%** |
| Classifier Flip (Twitter) | 5.7% | 5.7% | 4.3% |
| Classifier Flip (Kaggle) | 9.3% | 12.1% | 15.0% |
| Classifier Flip (News) | 24.3% | 25.0% | 31.4% |
| **Gap (Best Clf vs Human)** | **30.0 pp** | **29.3 pp** | **21.5 pp** |

### Golden Data Automated Metrics (140 samples)

| Metric | T5-Joint | T5-Control | BART-RL |
|--------|----------|------------|---------|
| Similarity | 0.915 | 0.881 | 0.864 |
| Edit Distance | 0.628 | 0.680 | 0.623 |
| BLEU vs Input | 0.236 | 0.187 | 0.241 |
| Paraphrase Score | 0.226 | 0.175 | 0.228 |
| Perplexity | 740.0 | 750.6 | 1794.4 |

---

## Multi-Classifier Comparison Study

### Why Multiple Classifiers?

We discovered classifiers disagree massively — using only one gives misleading results.

### Classifier Performance vs Human Ground Truth

**Averaged across 3 models (140 samples each):**

| Classifier | Accuracy | Precision | Recall | Cohen's κ |
|------------|----------|-----------|--------|-----------|
| RoBERTa-Twitter | 47.6% | 62.5% | 6.2% | +0.019 🟡 |
| Bert-Kaggle | 43.1% | 36.9% | 8.4% | **-0.075** 🔴 |
| RoBERTa-News | 53.6% | 64.2% | 31.9% | +0.104 🟡 |

### Full 9-Cell Breakdown (3 Models × 3 Classifiers)

| Model | Classifier | Clf Flip | Human Flip | Accuracy | κ |
|-------|------------|----------|------------|----------|---|
| T5-Joint | RoBERTa-Twitter | 5.7% | 54.3% | 48.6% | +0.044 |
| T5-Joint | Bert-Kaggle | 9.3% | 54.3% | 43.6% | -0.055 |
| T5-Joint | RoBERTa-News | 24.3% | 54.3% | 57.1% | **+0.179** |
| T5-Control | RoBERTa-Twitter | 5.7% | 54.3% | 47.1% | +0.017 |
| T5-Control | Bert-Kaggle | 12.1% | 54.3% | 40.7% | **-0.113** |
| T5-Control | RoBERTa-News | 25.0% | 54.3% | 50.7% | +0.055 |
| BART-RL | RoBERTa-Twitter | 4.3% | 52.9% | 47.1% | -0.005 |
| BART-RL | Bert-Kaggle | 15.0% | 52.9% | 45.0% | -0.058 |
| BART-RL | RoBERTa-News | 31.4% | 52.9% | 52.9% | +0.077 |

**4 out of 9 pairs show NEGATIVE κ (anti-correlation with humans)**

### Confusion Matrix (T5-Joint, RoBERTa-Twitter)

```
                          HUMAN GROUND TRUTH
                      Flipped (1)    Not Flipped (0)
                    ┌─────────────┬────────────────┐
CLASSIFIER          │             │                │
    Predicted 1     │      6      │       2        │
    (says flip)     │   (TP)      │     (FP)       │
                    ├─────────────┼────────────────┤
    Predicted 0     │     70      │      62        │
    (says no flip)  │   (FN)      │     (TN)       │
                    └─────────────┴────────────────┘

Miss Rate: 70/76 = 92% of real flips MISSED
```

---

## Subtype Analysis

### Distribution (2857 samples)

| Subtype | Count | Percentage |
|---------|-------|------------|
| sarcasm | 883 | 30.9% |
| irony | 614 | 21.5% |
| satire | 525 | 18.4% |
| overstatement | 401 | 14.0% |
| understatement | 330 | 11.6% |
| rhetorical_question | 104 | 3.6% |

### Performance by Subtype (T5-Joint, Big Test Data 2857 samples)

| Subtype | Count | Flip Rate | Similarity | BLEU | Edit Dist | Para Score |
|---------|-------|-----------|------------|------|-----------|------------|
| sarcasm | 883 | 7.5% | 0.867 | 0.213 | 0.61 | 0.193 |
| irony | 614 | 9.6% | 0.874 | 0.212 | 0.59 | 0.193 |
| satire | 525 | 6.1% | 0.888 | 0.222 | 0.57 | 0.204 |
| overstatement | 401 | 8.7% | 0.873 | 0.224 | 0.59 | 0.204 |
| understatement | 330 | 9.7% | 0.862 | 0.183 | 0.63 | 0.164 |
| rhetorical_question | 104 | 13.5% | 0.840 | 0.169 | 0.69 | 0.148 |

### Human Evaluation by Subtype (T5-Joint, Golden Data 140 samples)

| Subtype | N | Human Flip | Meaning Δ | Strict Success |
|---------|---|------------|-----------|----------------|
| sarcasm | 69 | 59.4% | 23.2% | 44.9% |
| irony | 30 | 50.0% | 6.7% | 50.0% |
| rhetorical_question | 14 | 50.0% | 0.0% | 50.0% |
| understatement | 11 | 54.5% | 18.2% | 36.4% |
| satire | 9 | 55.6% | 33.3% | 22.2% |
| overstatement | 7 | 28.6% | 0.0% | 28.6% |

### Classifier Error Rate by Subtype (T5-Joint, Golden Data)

| Subtype | N | Human Flip | Clf Flip (News) | Clf Miss Rate† | Primary Failure Mode |
|---------|---|------------|-----------------|----------------|----------------------|
| rhetorical_question | 14 | 50.0% | 14.3% | **71.4%** | Classifier failure |
| understatement | 11 | 54.5% | 27.3% | **50.0%** | Classifier failure |
| irony | 30 | 50.0% | 16.7% | **66.7%** | Classifier failure |
| satire | 9 | 55.6% | 11.1% | **80.0%** | Classifier failure |
| sarcasm | 69 | 59.4% | 33.3% | **43.9%** | Classifier failure |
| overstatement | 7 | 28.6% | 0.0% | **100%*** | **Model failure** |

†**Miss Rate** = percentage of human-labeled flips that the classifier failed to detect (False Negatives / Total Human Positives). Higher = worse classifier performance.

*⚠️ **Caution:** Sample sizes are small (N=7 to N=69). Overstatement (N=7) statistics are particularly unreliable. Overstatement has 100% miss rate, but this is because the **MODEL** fails to remove sarcasm (only 28.6% human flip rate) — classifier can't detect what didn't happen.

---

## Deep Analysis: Why These Results Occur

### Why Do ALL Classifiers Fail Against Human Judgment?

**Surface-level answer:** Domain mismatch (Twitter / Kaggle headline classifiers on news headlines).

**Deep analysis:**

The classifiers are trained for **sarcasm DETECTION** (is this text sarcastic?), but we're using them for **sarcasm REMOVAL verification** (did the output become non-sarcastic?). These are fundamentally different tasks.

The classifier never sees the INPUT-OUTPUT pair together — it judges each independently. It can't detect TRANSFORMATION, only static classification.

**Example:**
- Input: "Scientists Baffled By Man Who Exercises And Eats Well"
- Output: "Study confirms exercise and healthy diet improve health outcomes"
- Human: ✅ Sarcasm removed (the absurd premise is gone)
- Classifier: ❓ Both look like news headlines — can't detect the change

**World knowledge gap:**

Sarcasm often requires knowing what's ABSURD:
- "Google Unveils AI That Can Finally Make Eye Contact" — sarcastic because this is obviously exaggerated
- Classifier doesn't know what Google can/can't actually do
- Rewritten version might still "look" like tech news to the classifier

---

### Why Does T5-Joint Beat T5-Control on Meaning Preservation?

| Metric | T5-Joint | T5-Control | Δ |
|--------|----------|------------|---|
| Human Meaning Change | **16.4%** | 25.0% | -8.6 pp |
| Human Strict Success | **43.6%** | 39.3% | +4.3 pp |

**The strategy prefix forces DECOMPOSITION before generation.**

Joint model first identifies the sarcasm type: "[EXAGGERATION]" or "[IRONY]" or "[RHETORICAL_QUESTION]"

This tells the model:
1. **WHAT aspect to change** — the specific sarcasm mechanism
2. **WHAT to preserve** — everything else

Different sarcasm types need different rewriting strategies:
- Exaggeration → tone down the claim
- Rhetorical question → convert to statement
- Irony → state the actual intended meaning

**Without the prefix (Control):**
- Model must simultaneously figure out: What's sarcastic? How to fix it? What to keep?
- More cognitive load → more mistakes → more meaning drift

**Analogy:** It's like asking someone to "edit this document" vs "fix the grammar errors in this document". The second is more constrained and leads to more targeted changes.

---

### Why Does BART-RL Destroy Meaning (40.7% meaning change)?

**The RL reward function creates a perverse incentive:**

Reward = maximize P(non-sarcastic) + maintain ROUGE-L overlap

The model learns a shortcut:
- DELETING sarcastic words reduces P(sarcastic) ✅
- Keeping SOME words maintains ROUGE-L overlap ✅
- But deletion ≠ rewriting!

**Example:**
- Input: "Area Man Proud Of Completely Average Achievement"
- BART-RL output: "Man achieves something"
- Classifier: Happy (no sarcasm markers)
- ROUGE-L: Some overlap (Man, achieves)
- Human: ❌ Meaning destroyed — the IRONY about pride in averageness is gone, not rewritten

**This is reward hacking:** Models find shortcuts that optimize the reward function without achieving the actual goal. Deletion is "easier" than genuine rewriting.

---

### Why Do LLaMA and BART-CE Completely Rewrite (Similarity 0.61–0.66)?

These models show: LLM Sarcasm = 5.0/5, LLM Meaning = 2.9–3.3/5

**This isn't style transfer — it's GENERATION.**

| Model | Similarity | Edit Dist | BLEU | LLM Meaning |
|-------|------------|-----------|------|-------------|
| llama_3_2_1b | 0.656 | 0.948 | 0.013 | 3.34 |
| bart_base_ce | 0.636 | 0.923 | 0.023 | 3.52 |
| bart_base_ce_rl | 0.609 | 0.928 | 0.021 | 2.90 |

**Why BART-CE fails:**
- Trained on parallel sarcastic → non-sarcastic pairs with cross-entropy loss
- Model learns: "given a sarcastic topic, generate a non-sarcastic headline about that topic"
- No explicit constraint saying "preserve the specific claims/facts"
- Result: topically related but factually different output

**Why LLaMA is even more extreme:**
- Edit distance 0.95 = almost complete rewrite
- BLEU 0.01 = virtually no word overlap
- It's essentially generating a NEW headline inspired by the input
- The model has so much generation capacity that it "forgets" to preserve the original content

---

### Why Do Ablation Models Show Minimal Difference?

| Ablation | Similarity | LLM Meaning | LLM Sarcasm |
|----------|------------|-------------|-------------|
| Without irony | 0.882 | 4.60 | 3.86 |
| Without overstatement | 0.885 | 4.58 | 3.28 |
| Without rhetorical_q | 0.881 | 4.42 | 3.18 |
| Without sarcasm | 0.883 | 4.24 | 3.32 |
| Without satire | 0.880 | 4.52 | 3.18 |
| Without understatement | 0.881 | 4.44 | 3.52 |

**Sarcasm subtypes share underlying mechanisms:**

- Exaggeration and overstatement both use HYPERBOLE
- Irony and sarcasm both use CONTRADICTION between literal and intended meaning
- Rhetorical questions often contain embedded irony

The model learns GENERAL sarcasm patterns from 5 subtypes that transfer to the 6th. Removing one subtype doesn't cripple the model because the remaining subtypes provide sufficient coverage.

**This is actually a positive finding about generalization!**

---

### Why Do Different Subtypes Have Different Classifier Miss Rates?

#### Satire — Highest Miss Rate (80.0%)

**Satire MIMICS legitimate news format with ABSURD content.**

Example: "Congress Votes To Continue Doing Nothing"
- Looks like: Real political news headline
- Actually: Exaggerated commentary on political dysfunction

**Why classifier fails:**
- Format deliberately mimics real news (The Onion copies AP style)
- Requires CULTURAL CONTEXT to know what's ridiculous
- "Scientists Confirm Earth Still Round" — classifier can't know this is satirical vs real news

**Why miss rate is highest:** After rewriting, satirical content often becomes indistinguishable from legitimate news. The classifier can't detect that sarcasm was removed because both look like normal headlines.

#### Rhetorical Questions — High Miss Rate (71.4%)

**The sarcasm lives in the IMPLICATION, not the words.**

Example: "Who Actually Believes This Works?"
- Literal meaning: A question asking who believes something
- Sarcastic meaning: "Nobody should believe this"
- When rewritten to: "Many people doubt the effectiveness of this"

**Why classifier fails:**
- Trained mostly on declarative statements, not questions
- Questions have different syntax patterns
- The ANSWER is implied, not stated — classifier can't "see" it
- Even after rewriting, the topic remains the same, so classifier thinks nothing changed

**Deep insight:** Rhetorical questions encode sarcasm in PRAGMATICS (what's implied), not SEMANTICS (what's literally said). Classifiers only see semantics.

#### Irony — High Miss Rate (66.7%)

**Irony says the OPPOSITE of what's meant — with NO surface markers.**

Example: "What A Great Day For Democracy"
- Literal: Positive statement about democracy
- Ironic meaning: Today was terrible for democracy

**Why classifier fails:**
- No exaggeration ("great" is a normal positive word)
- No question marks
- No caps or unusual punctuation
- Words are genuinely positive — classifier sees "good sentiment"
- The contradiction is CONTEXTUAL, not lexical

**Deep insight:** Irony creates meaning through CONTRADICTION with context. Classifiers see text in isolation, missing the situational contradiction that makes it ironic.

#### Understatement — Moderate Miss Rate (50.0%)

**Understatement uses NEUTRAL words to describe EXTREME situations.**

Example: "Minor Setback Causes Some Concern"
- Reality: Major disaster causes widespread panic
- The sarcasm is the GAP between words and reality

**Why classifier fails:**
- Sees neutral, non-extreme language
- No exaggeration words to trigger detection
- Requires WORLD KNOWLEDGE to know this is understated
- "Some concern" about a plane crash → classifier thinks this is just normal news

**Deep insight:** Understatement detection requires knowing what the "normal" or "proportionate" response would be. Classifiers don't have this calibrated sense of proportion.

#### Generic Sarcasm — Lowest Miss Rate (43.9%)

**Generic sarcasm has the most EXPLICIT markers.**

Example: "What An Absolutely Brilliant Idea"
- Markers: "Absolutely", exaggerated praise, excessive superlatives
- Pattern: Excessive positivity about something negative

**Why classifier does better:**
- MOST training examples are this type
- Has learnable surface patterns (exaggeration words, excessive superlatives)
- Patterns like "totally", "definitely", "what a surprise" appear in training data

**Deep insight:** Generic sarcasm evolved as a category BECAUSE it has detectable patterns. The classifier was trained on data where these patterns exist.

#### Overstatement — Anomalous (100% Miss Rate, but MODEL Failure!)

**This is NOT classifier success — it's MODEL failure.**

| Metric | Overstatement | Other Subtypes |
|--------|---------------|----------------|
| Human Flip Rate | **28.6%** | 50–59% |
| N | 7 | 9–69 |

**Three anomalies:**
1. Lowest human flip rate (28.6% vs ~50-60% for others) — model fails to remove sarcasm
2. All 3 classifiers report 0% — outputs still "look" sarcastic
3. Tiny sample size (N=7) — statistics unreliable

**Why the MODEL fails on overstatement:**

Overstatement is HARD to rewrite without losing the core claim:

Example:
- Input: "This Is Literally The Most Important Discovery In Human History"
- The claim IS the exaggeration — there's nothing underneath to preserve
- If you remove exaggeration: "Scientists make a discovery" — but is this even the same story?
- Hard to know what's the "appropriate" level of enthusiasm without knowing the actual importance

**The model's dilemma:**
1. Keep the exaggeration → Still sarcastic (human says not flipped)
2. Remove too much → Meaning lost
3. Find the "right" level → Requires knowing actual importance of event

**Key insight:** Overstatement may need external knowledge about "appropriate" levels of emphasis — something the model doesn't have.

---

### Summary: Why Subtypes Have Different Classifier Miss Rates

| Subtype | Miss Rate† | Why Classifier Fails | What Would Help |
|---------|-----------|---------------------|-----------------|
| overstatement | 100%* | *Model fails to remove, not classifier | External knowledge of "appropriate" emphasis |
| satire | 80.0% | Mimics real news format | Cultural/current events knowledge |
| rhetorical_question | 71.4% | Sarcasm in implication, not words | Pragmatic inference |
| irony | 66.7% | Contradiction with context, no markers | Situational understanding |
| understatement | 50.0% | Neutral words, needs world knowledge | Proportionality reasoning |
| sarcasm (generic) | 43.9% | Has surface markers — classifier learned these | More training data |

†Miss Rate = percentage of human-labeled flips that the classifier failed to detect. *Overstatement is anomalous: 100% miss rate occurs because the MODEL fails to remove sarcasm (only 28.6% human flip rate), not because the classifier is bad.

---

## Key Findings

### Finding 1: All Classifiers Are Unreliable

| Evidence | Value |
|----------|-------|
| Average κ (best classifier) | +0.104 |
| Average κ (worst classifier) | -0.075 |
| Negative κ pairs | 4/9 (44%) |
| Human agreement κ | >0.8 |

**Root cause:** Classifiers detect sarcasm PRESENCE, not sarcasm REMOVAL. They judge outputs in isolation, missing the transformation.

### Finding 2: Massive Classifier Disagreement

| Statistic | Value |
|-----------|-------|
| Average spread (same model) | 28.2 pp |
| Maximum spread | 33.4 pp |
| Models with >30pp spread | 12/14 |

**Root cause:** Different classifiers learned different surface patterns from different training data.

### Finding 3: Trade-off Between Sarcasm Removal and Meaning

| Model Type | LLM Sarcasm | LLM Meaning | Similarity |
|------------|-------------|-------------|------------|
| Conservative (T5) | 3.7 | 4.7 | 0.87 |
| Aggressive (LLaMA) | 5.0 | 3.3 | 0.66 |

**Root cause:** Aggressive rewriting removes more sarcasm markers but also removes content. The safest way to "remove sarcasm" is to delete everything — which destroys meaning.

### Finding 4: T5-Joint Best Overall

| Metric | T5-Joint | T5-Control | BART-RL |
|--------|----------|------------|---------|
| Human Meaning Change | **16.4%** | 25.0% | 40.7% |
| Human Strict Success | **43.6%** | 39.3% | 34.3% |
| Similarity | 0.915 | 0.881 | 0.864 |
| LLM Meaning | **4.82** | 4.64 | 4.26 |

**Root cause:** Strategy prefix forces task decomposition — identify what's sarcastic BEFORE rewriting.

### Finding 5: Ablation Models Show Minimal Difference

| Metric | Range Across 6 Ablations |
|--------|--------------------------|
| Similarity | 0.880 – 0.885 |
| LLM Meaning | 4.24 – 4.60 |
| LLM Sarcasm | 3.18 – 3.86 |
| Paraphrase Score | 0.207 – 0.217 |

**Root cause:** Sarcasm subtypes share underlying mechanisms (hyperbole, contradiction, absurdity). Knowledge transfers across subtypes.

### Finding 6: LLaMA/BART-CE Fail at Style Transfer

| Model | Similarity | Edit Dist | LLM Meaning | Problem |
|-------|------------|-----------|-------------|---------|
| llama_3_2_1b | 0.656 | 0.948 | 3.34 | Complete rewrite |
| bart_base_ce | 0.636 | 0.923 | 3.52 | Complete rewrite |
| bart_base_ce_rl | 0.609 | 0.928 | 2.90 | Complete rewrite |

**Root cause:** These models have too much generative capacity — they "forget" to preserve the original and generate new content instead.

### Finding 7: Different Subtypes Have Different Failure Modes

| Subtype | Clf Miss Rate | Failure Mode |
|---------|---------------|--------------|
| overstatement | 100%* | MODEL fails to remove — classifier can't detect what didn't happen |
| satire | 80.0% | Format mimics real news |
| rhetorical_question | 71.4% | Pragmatic sarcasm — classifier can't see implication |
| irony | 66.7% | Contextual contradiction — no surface markers |
| understatement | 50.0% | World knowledge required |
| sarcasm (generic) | 43.9% | Surface markers — classifier learned these |

*Overstatement has only 28.6% human flip rate (MODEL failure), so miss rate is 100% by default.

**Root cause:** Sarcasm isn't one phenomenon — it's many. Different mechanisms require different detection strategies.

---

## Conclusions

### Summary Table

| Question | Answer | Evidence | Why? |
|----------|--------|----------|------|
| Are classifiers reliable? | **NO** | κ = -0.08 to +0.10 | Detection ≠ removal verification |
| Best model? | **T5-Joint** | 43.6% strict success | Strategy prefix → decomposition |
| Worst models? | **LLaMA, BART-CE** | Similarity < 0.66 | Generation not transformation |
| Do ablations matter? | **Minimal** | All similar | Subtypes share mechanisms |
| Hardest subtype? | **Satire** | 80% miss rate | Mimics real news format |
| Easiest subtype? | **Generic sarcasm** | 44% miss rate | Has surface markers |
| Anomalous subtype? | **Overstatement** | MODEL fails (28.6% flip) | Hard to rewrite without world knowledge |

### Metric Recommendations

| Metric | Primary Use | Limitation | When to Trust |
|--------|-------------|------------|---------------|
| Similarity | Meaning preservation | Doesn't detect copying | Always check with BLEU |
| BLEU vs Input | Detect copying | High BLEU ≠ quality | Use with similarity |
| Edit Distance | Rewrite extent | Doesn't indicate quality | Compare across models |
| Perplexity | Fluency check | Doesn't measure task success | Flag outliers only |
| LLM Judge | Holistic quality | Expensive, potential bias | Sample evaluation |
| Paraphrase Score | Genuine rewriting | Exploratory metric | Use for ranking |
| Flip Rate | Sarcasm removal | **UNRELIABLE** | Use all 3 classifiers + human |

### The Takeaway

> "Automated flip rate is NOT a valid primary metric. Classifiers detect sarcasm presence, not removal — they judge outputs in isolation, missing the transformation. Different sarcasm subtypes fail for different reasons: rhetorical questions encode sarcasm in implication, understatement requires world knowledge, irony needs contextual understanding. Human evaluation reveals what classifiers cannot see. Use all 7 metrics together."

---

## Files

```
scripts/
├── eval_pipeline.py          # Main 7-metric pipeline (--multi_classifier)
├── batch_eval.py             # Batch evaluation (14 models)
├── analyze_golden_results.py # Classifier vs human comparison
└── clean_golden_data.py      # Standardize annotations

data/golden/cleaned/          # 140 samples × 3 models (human annotated)
model_outputs_clean/          # 2857 samples × 14 models
results/                      # All evaluation outputs
```

---

## Citation

```
Project LLMao: Multi-Metric Evaluation of Sarcasm Style Transfer
CS4248 Team 14, National University of Singapore
April 2026

Key Finding: Automated sarcasm classifiers (κ = -0.08 to +0.10) fail 
against human evaluation (κ > 0.8) because they detect sarcasm presence, 
not removal. Different subtypes fail for different reasons — rhetorical 
questions encode sarcasm in implication, understatement requires world 
knowledge, irony needs contextual understanding. Multi-metric evaluation 
with human validation is essential.
```
