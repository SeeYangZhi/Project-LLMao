# Evaluation Pipeline Documentation

## Project LLMao — CS4248 Team 14

**Task:** Sarcasm Style Transfer (convert sarcastic headlines to non-sarcastic while preserving meaning)

---

## Table of Contents

1. [Overview](#overview)
2. [The 7-Metric Pipeline](#the-7-metric-pipeline)
3. [Human Evaluation Protocol](#human-evaluation-protocol)
4. [Classifier Comparison Study](#classifier-comparison-study)
5. [Key Findings](#key-findings)
6. [Conclusions](#conclusions)

---

## Overview

### The Challenge

Evaluating sarcasm style transfer is fundamentally difficult because:

1. **No ground truth** — there's no single "correct" non-sarcastic version of a sarcastic headline
2. **Dual objectives** — must remove sarcasm AND preserve meaning
3. **Metrics are gameable** — models can fool classifiers without genuinely removing sarcasm

### Our Solution

We developed a **multi-faceted evaluation approach**:

- **7 automated metrics** covering different aspects of quality
- **Human evaluation** on 140 golden samples with 2 annotators per model
- **Classifier comparison study** testing 3 different sarcasm detectors

---

## The 7-Metric Pipeline

### 1. Flip Rate (Sarcasm Classifier)

**Tool:** `cardiffnlp/twitter-roberta-base-irony`

**What it measures:** Does the classifier think sarcasm was removed?

**How it works:**
- Runs classifier on input and output
- `hard_flipped = 1` if input was ironic AND output is non-ironic
- `flip_delta` = P(ironic|input) - P(ironic|output)

**Why we use it:** Standard metric for style transfer success

**Limitations:** 
- Trained on Twitter, not news headlines (domain mismatch)
- Detects surface markers, not semantic sarcasm
- **Anti-correlated with human judgment (κ = -0.07 to -0.19)**

---

### 2. Semantic Similarity

**Tool:** `sentence-transformers/all-MiniLM-L6-v2`

**What it measures:** Is the core meaning preserved?

**How it works:**
- Embeds input and output as 384-dim vectors
- Computes cosine similarity

**Why we use it:** Captures semantic preservation beyond word overlap

**Interpretation:**
- 0.95+ = Nearly identical meaning (possibly just paraphrased)
- 0.85-0.95 = Good meaning preservation
- < 0.70 = Significant meaning drift

**Limitations:** May not capture subtle meaning changes that humans notice

---

### 3. Perplexity

**Tool:** GPT-2

**What it measures:** Is the output fluent, natural English?

**How it works:**
- Computes cross-entropy loss on output text
- Lower = more natural language

**Why we use it:** Catches degenerate outputs (repetitions, truncations, nonsense)

**Interpretation:**
- < 100 = Very fluent
- 100-500 = Normal
- 500-1000 = Somewhat disfluent
- > 1000 = Problematic (we filter > 10000)

---

### 4. BLEU vs Input

**Tool:** `sacrebleu`

**What it measures:** N-gram overlap between output and INPUT (not reference)

**Why this matters:** Unlike standard BLEU (vs reference), this detects COPYING.

**Interpretation:**
- High BLEU (> 0.3) + High Similarity = Paraphrasing (just minor edits)
- Low BLEU (< 0.1) + High Similarity = Genuine rewriting (good!)
- Low BLEU + Low Similarity = Meaning lost (bad!)

---

### 5. Edit Distance

**Tool:** Word-level Levenshtein distance, normalized to [0, 1]

**What it measures:** How many word operations (insert/delete/replace) to transform input → output?

**Why we use it:** Complements BLEU by capturing structural changes

**Interpretation:**
- 0.0 = Identical
- 0.1-0.3 = Minor edits (punctuation, casing)
- 0.5-0.7 = Moderate rewriting
- 0.9+ = Complete rewrite

**Model patterns:**
- T5: 0.57-0.68 (conservative, minimal edits)
- BART-CE/LLaMA: 0.92-0.95 (aggressive rewriting)

---

### 6. LLM-as-Judge

**Tool:** Gemini 2.5 Flash

**What it measures:** Holistic quality assessment on 3 dimensions:
- `sarcasm_removed` (1-5): Is the output non-sarcastic?
- `meaning_preserved` (1-5): Is the core meaning intact?
- `fluency` (1-5): Is it natural English?

**Why we use it:** Captures nuances that rule-based metrics miss

**Limitations:**
- Expensive, slow
- Potential biases in LLM judgment
- Black box — hard to interpret

---

### 7. Paraphrase Score (Exploratory)

**Formula:** `similarity × (1 - BLEU_vs_input)`

**What it measures:** Did the model genuinely rewrite (high similarity, low copying)?

**Interpretation:**
- High score (> 0.5) = Good semantic preservation with actual rewriting
- Low score (< 0.1) = Either meaning lost OR just copied input

**Note:** This is exploratory — helps identify models that "game" metrics by surface edits

---

## Human Evaluation Protocol

### Setup

- **140 samples** per model (stratified by sarcasm subtype)
- **3 models evaluated:** T5-Joint, T5-Control, BART-RL
- **2 independent annotators** per model
- **Labels collected:**
  - `sarcasm_removed`: Binary (Y/N) — is the output non-sarcastic?
  - `meaning_change`: Binary (Y/N) — did the meaning change significantly?

### Annotators

| Model | Annotators |
|-------|------------|
| T5-Joint | Angel + Camille |
| T5-Control | Angel + Camille |
| BART-RL | Nguyen + Andrew |

### Inter-Annotator Agreement

| Model | Raw Agreement | Cohen's κ | Interpretation |
|-------|---------------|-----------|----------------|
| T5-Joint | 92.1% | 0.839 | Excellent |
| T5-Control | 94.3% | 0.883 | Excellent |
| BART-RL | 94.3% | 0.884 | Excellent |

**All κ > 0.8 = Excellent agreement** → Human labels are reliable ground truth

### Derived Metrics

- `human_flipped_strict`: Both annotators agree sarcasm was removed
- `human_flipped_lenient`: At least one annotator says sarcasm was removed
- `human_strict_success`: Flipped AND meaning preserved

---

## Classifier Comparison Study

### Motivation

Our initial classifier (RoBERTa-Twitter-Irony) showed **negative κ** with human judgment. We asked: Is this a problem with one classifier, or a fundamental limitation of automated sarcasm detection?

### Classifiers Tested

| Classifier | Architecture | Training Data |
|------------|--------------|---------------|
| RoBERTa-Twitter-Irony | RoBERTa | Twitter |
| DistilBERT-Reddit | DistilBERT | Reddit |
| BERT-Sarcasm-News | RoBERTa | News Headlines |

### Results

#### By Classifier (averaged across models)

| Classifier | Flip Rate | Accuracy | Precision | Recall | κ |
|------------|-----------|----------|-----------|--------|---|
| BERT-Sarcasm-News | 33.3% | 55.2% | 63.9% | 39.4% | **+0.128** 🟡 |
| DistilBERT-Reddit | 86.2% | 54.8% | 55.0% | 88.1% | +0.043 🟡 |
| RoBERTa-Twitter-Irony | 21.4% | 41.0% | 36.9% | 15.0% | **-0.132** 🔴 |

#### Full Comparison Table

| Model | Classifier | Clf Flip | Human Flip | Accuracy | κ |
|-------|------------|----------|------------|----------|---|
| T5-Joint | RoBERTa-Twitter | 24.3% | 54.3% | 44.3% | -0.067 🔴 |
| T5-Joint | DistilBERT-Reddit | 83.6% | 54.3% | 50.7% | -0.046 🔴 |
| T5-Joint | BERT-News | 30.7% | 54.3% | 59.3% | **+0.212** 🟢 |
| T5-Control | RoBERTa-Twitter | 21.4% | 54.3% | 40.0% | -0.144 🔴 |
| T5-Control | DistilBERT-Reddit | 83.6% | 54.3% | 56.4% | +0.075 🟡 |
| T5-Control | BERT-News | 31.4% | 54.3% | 51.4% | +0.059 🟡 |
| BART-RL | RoBERTa-Twitter | 18.6% | 52.9% | 38.6% | -0.186 🔴 |
| BART-RL | DistilBERT-Reddit | 91.4% | 52.9% | 57.1% | +0.100 🟡 |
| BART-RL | BERT-News | 37.9% | 52.9% | 55.0% | +0.112 🟡 |

### Key Observations

1. **Training domain matters:**
   - News-trained classifier (κ = +0.128) outperforms Twitter-trained (κ = -0.132)
   - Domain match helps, but is still insufficient

2. **Even the best classifier fails:**
   - Best κ = 0.212 (T5-Joint with BERT-News) = "fair agreement"
   - Average κ across all = +0.013 ≈ random chance

3. **4 out of 9 classifier-model pairs show negative κ**
   - Anti-correlation = classifier is systematically wrong

4. **Different failure modes:**
   - RoBERTa-Twitter: Low flip rate (21%), misses real flips
   - DistilBERT-Reddit: High flip rate (86%), many false positives

---

## Key Findings

### Finding 1: Classifier is Anti-Correlated with Human Judgment

**Evidence (RoBERTa-Twitter-Irony):**

| Model | Classifier Flip | Human Flip | κ |
|-------|-----------------|------------|---|
| T5-Joint | 24.3% | 54.3% | -0.067 |
| T5-Control | 21.4% | 54.3% | -0.144 |
| BART-RL | 18.6% | 52.9% | -0.186 |

**Confusion Matrix (T5-Joint):**

```
                      HUMAN
                Flipped   Not Flipped
            ┌──────────┬─────────────┐
CLASSIFIER  │          │             │
  Flipped   │    16    │     18      │ ← 18 False Positives
            ├──────────┼─────────────┤
  Not Flip  │    60    │     46      │ ← 60 False Negatives
            └──────────┴─────────────┘
```

**Why:** Classifier detects surface markers (lowercase, punctuation), not semantic sarcasm. Models learn to "game" it with surface edits.

---

### Finding 2: Surface Edits Fool the Classifier

**10% of outputs are just lowercase/punctuation changes:**

| Input | Output | Classifier | Human |
|-------|--------|------------|-------|
| "Google presses play on 30-second Gemini musical slop generator" | "google presses play on 30-second Gemini musical slop generator" | FLIPPED ✗ | NOT FLIPPED ✓ |

The classifier thinks lowercase = sarcasm removed!

---

### Finding 3: Joint Beats Control (Invisible to Classifier)

| Metric | T5-Joint | T5-Control | Δ |
|--------|----------|------------|---|
| Meaning Change | **16.4%** | 25.0% | -8.6 pp |
| Strict Success | **43.6%** | 39.3% | +4.3 pp |
| Classifier Accuracy | 44.3% | 40.0% | ~Same |

**Why Joint wins:** Strategy prefix forces model to decompose the task — identify the sarcasm mechanism BEFORE rewriting, so it knows what to preserve.

**Critical insight:** Classifier CANNOT see this difference. Only human evaluation reveals Joint is better.

---

### Finding 4: BART-RL Destroys Meaning

| Model | Human Flip | Meaning Change | Strict Success |
|-------|------------|----------------|----------------|
| T5-Joint | 54.3% | 16.4% | 43.6% |
| T5-Control | 54.3% | 25.0% | 39.3% |
| BART-RL | 52.9% | **40.7%** | 34.3% |

**Why:** RL reward function incentivizes deletion. Removing words reduces P(sarcastic) while maintaining ROUGE-L overlap. The model learned: deletion is rewarded.

---

### Finding 5: Subtype Difficulty Varies

| Subtype | Classifier Error Rate | Why? |
|---------|----------------------|------|
| rhetorical_question | 85.7% | Explicit markers but classifier still fails |
| understatement | 81.8% | Subtle, requires world knowledge |
| irony | 73.3% | Semantic contradiction, no surface markers |
| satire | 55.6% | Requires cultural context |
| sarcasm | 53.6% | Most common, classifier somewhat learned |

---

## Conclusions

### For Researchers

1. **Automated sarcasm classifiers cannot reliably evaluate style transfer outputs.**
   - Average κ across 3 classifiers = +0.013 (random chance)
   - Even the best classifier (news-trained) only achieves κ = 0.128 ("slight agreement")

2. **Human evaluation is necessary** for this task.
   - Inter-annotator κ > 0.8 = reliable ground truth
   - Reveals quality differences invisible to automated metrics

3. **Domain match helps but isn't enough.**
   - News-trained classifier better than Twitter-trained
   - But still fails to capture semantic sarcasm removal

### For Practitioners

1. **Use multiple metrics** — no single metric captures everything
2. **Validate with human evaluation** — at least a sample
3. **Be skeptical of automated flip rates** — they can be gamed

### Takeaway

> "Automated metrics are necessary but not sufficient. Models learn to game classifiers with surface edits. Human evaluation reveals what classifiers cannot see."

---

## Files

```
scripts/
├── eval_pipeline.py          # Main 7-metric pipeline
├── clean_golden_data.py      # Standardize human annotations
├── analyze_golden_results.py # Classifier vs human analysis
└── compare_classifiers.py    # Multi-classifier comparison

data/golden/
├── raw/                      # Original human annotations
└── cleaned/                  # Standardized format

results/golden/
├── *_results.csv             # Automated metrics
├── *_merged.csv              # Human + automated merged
├── *_subtype_analysis.csv    # Per-subtype breakdown
├── classifier_comparison.csv # Multi-classifier results
└── summary.csv               # Overall comparison
```

---

## Citation

If you use this evaluation framework, please cite:

```
Project LLMao: Sarcasm Style Transfer Evaluation
CS4248 Team 14, NUS
April 2026
```
