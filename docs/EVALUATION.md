# Evaluation Pipeline Documentation

**Project:** CS4248 Team 14 "Project LLMao"
**Last Updated:** 2026-04-12

---

## 1. Evaluation Goals

We evaluate sarcasm-to-neutral rewriting on **three core properties**:

| Property | Question | Why It Matters |
|----------|----------|----------------|
| **Sarcasm Removal** | Did the sarcasm go away? | Primary task objective |
| **Meaning Preservation** | Is the core message intact? | Useless if meaning is lost |
| **Fluency** | Is the output grammatical? | Must be readable |

**The Challenge:** No single metric captures all three. A model can:
- Remove sarcasm by deleting everything (high flip rate, zero meaning)
- Preserve meaning by copying input (high similarity, no sarcasm removal)
- Generate fluent nonsense (low perplexity, wrong content)

Our pipeline uses **7 complementary metrics** to detect these failure modes.

---

## 2. Metrics Overview

| # | Metric | Measures | Range | Good Score |
|---|--------|----------|-------|------------|
| 1 | **Flip Rate** | Sarcasm removal | 0-1 | Higher = more sarcasm removed |
| 2 | **Flip Delta** | Confidence shift | -1 to +1 | Higher = stronger tone change |
| 3 | **Semantic Similarity** | Meaning preservation | 0-1 | Higher = meaning kept |
| 4 | **Perplexity** | Fluency | 0-∞ | Lower = more fluent |
| 5 | **BLEU vs Input** | Surface-level copying | 0-1 | Context-dependent |
| 6 | **Edit Distance** | Amount of rewriting | 0-1 | Context-dependent |
| 7 | **LLM-as-Judge** | Holistic human-like rating | 1-5 | Higher = better |

---

## 3. Metric Details

### 3.1 Sarcasm Flip Rate

**Model:** `cardiffnlp/twitter-roberta-base-irony`

**What it computes:**
- `input_prob`: P(sarcastic | input)
- `output_prob`: P(sarcastic | output)
- `hard_flipped`: 1 if input was sarcastic AND output is non-sarcastic
- `flip_delta`: input_prob - output_prob (how much the tone shifted)

**Why we need it:**  
Primary metric for task success. If the classifier still detects sarcasm, the model failed.

**Limitation discovered:**  
The classifier detects **surface features** (lowercase, punctuation, length) rather than semantic sarcasm. This causes:
- False positives: "GREAT news!" → "great news" (just lowercased, classifier says flipped)
- False negatives: Actual sarcasm removal missed because output still has "suspicious" words

**Evidence from our data:**

| Metric | Value |
|--------|-------|
| Classifier-Human Cohen's κ | -0.07 to -0.19 (ANTI-CORRELATED) |
| Classifier flip rate | ~19-24% across all models |
| Human flip rate | ~53-54% across all models |

**Conclusion:** Flip rate is necessary but not sufficient. Must validate with human evaluation.

---

### 3.2 Semantic Similarity

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

**What it computes:**  
Cosine similarity between input and output sentence embeddings.

**Why we need it:**  
Ensures the model didn't change the topic or lose the core message.

**Interpretation:**

| Score | Meaning |
|-------|---------|
| > 0.9 | Very similar (possibly just paraphrased) |
| 0.7-0.9 | Good meaning preservation |
| 0.5-0.7 | Significant rewording |
| < 0.5 | Topic drift or hallucination |

**Limitation:**  
High similarity doesn't mean the OUTPUT is correct — it just means input and output are about the same topic. A model could keep the sarcasm and still have high similarity.

---

### 3.3 Perplexity (Fluency)

**Model:** `gpt2`

**What it computes:**  
Cross-entropy loss of the output under GPT-2's language model. Lower = more probable = more fluent.

**Why we need it:**  
Detects broken, repetitive, or ungrammatical outputs.

**Interpretation:**

| Score | Meaning |
|-------|---------|
| < 100 | Very fluent (common phrases) |
| 100-500 | Normal fluency |
| 500-1000 | Slightly awkward |
| > 1000 | Likely broken (filtered from averages) |

**Note:** We filter samples with perplexity > 10000 as outliers (usually empty or corrupted outputs).

---

### 3.4 BLEU vs Input

**Library:** `nltk.translate.bleu_score`

**What it computes:**  
N-gram overlap between OUTPUT and INPUT (not reference).

**Why vs INPUT, not reference?**  
We don't have gold-standard reference outputs. Instead, we use BLEU to detect **copying behavior**:
- High BLEU vs input = model kept most of the same words
- Low BLEU vs input = model genuinely rewrote the text

**Why we need it:**  
Combined with similarity, it distinguishes:
- **Paraphrasing** (high similarity + high BLEU): Same meaning, same words → model just copied
- **Genuine rewriting** (high similarity + low BLEU): Same meaning, different words → good!

---

### 3.5 Edit Distance (Normalized)

**Algorithm:** Word-level Levenshtein distance, normalized by max length.

**What it computes:**  
Proportion of words that were added, deleted, or changed.

**Interpretation:**

| Score | Meaning |
|-------|---------|
| 0.0 | Identical (no edits) |
| 0.3-0.5 | Moderate editing |
| 0.5-0.7 | Substantial rewriting |
| > 0.8 | Almost completely rewritten |

**Why we need it:**  
Edit distance is model-agnostic and well-established. Unlike BLEU, it counts insertions/deletions explicitly.

**Our findings:**

| Model | Edit Distance | Behavior |
|-------|---------------|----------|
| T5 models | 0.57-0.68 | Conservative editing |
| BART-CE/RL | 0.62-0.93 | Variable rewriting |
| LLaMA | 0.95 | Complete rewrite |

---

### 3.6 LLM-as-Judge

**Model:** Gemini 2.5 Flash (via API)

**What it computes:**  
For a batch of 50 samples, the LLM rates each (input, output) pair on:
- `sarcasm_removed`: 1-5 (5 = completely non-sarcastic)
- `meaning_preserved`: 1-5 (5 = identical meaning)
- `fluency`: 1-5 (5 = perfectly fluent)

**Why we need it:**  
Provides holistic human-like judgment that considers context, world knowledge, and pragmatics — things rule-based metrics miss.

**Validation:**  
We compute Cohen's κ between LLM judge and classifier to check agreement. In our experiments, κ ranged from -0.09 to 0.32, indicating the LLM and classifier see different things.

**Limitation:**
- Only run on 50 samples per model (cost/time constraint)
- LLM may have its own biases
- Not a replacement for human evaluation

---

### 3.7 Paraphrase Score (Exploratory)

**Formula:** `paraphrase_score = similarity × BLEU_vs_input`

**What it detects:**  
When BOTH similarity AND BLEU are high, the model likely just paraphrased (copied with minor edits) rather than genuinely rewrote.

**Why multiply?**

| Similarity | BLEU | Product | Interpretation |
|------------|------|---------|----------------|
| High | High | **High** | Paraphrasing (bad) — same words, same meaning |
| High | Low | Low | Genuine rewrite (good) — different words, same meaning |
| Low | High | Low | Rare case |
| Low | Low | Low | Topic drift or heavy editing |

**Concrete Examples from Our Data:**

**Example 1: PARAPHRASING (High Score = Bad)**

| Field | Value |
|-------|-------|
| Input | "Black Half Of Tiger Woods Tased By Cops After Asian Half Crashes Car" |
| Output | "black half of tiger woods tased by cops after Asian half crashes car" |
| Similarity | 1.00 |
| BLEU | 1.00 |
| Paraphrase Score | 1.00 |
| Human judgment | NOT FLIPPED (still sarcastic, just lowercased) |

**Example 2: GENUINE REWRITE (Low Score = Good)**

| Field | Value |
|-------|-------|
| Input | "Surprise! Big Tech has been a bit rubbish at enforcing Australia's kids social media ban" |
| Output | "big tech is failing to enforce Australia's children social media ban" |
| Similarity | 0.66 |
| BLEU | 0.04 |
| Paraphrase Score | 0.026 |
| Human judgment | FLIPPED, meaning preserved ✓ |

**Status:** This metric is exploratory. We report it but do not claim it as a contribution. The key insight is that similarity alone or BLEU alone can miss paraphrasing — the combination helps.

---

## 4. Human Evaluation (Ground Truth)

Because automated metrics have limitations, we conducted human evaluation on 140 samples.

### 4.1 Protocol

- **Annotators:** 2 per model (Angel+Camille for T5, Nguyen+Andrew for BART-RL)
- **Questions:**
  - Is the output non-sarcastic? (Yes=1, No=0)
  - Did the meaning change? (Yes=1, No=0)
- **Consensus:** Strict (both agree) and Lenient (either agrees)

### 4.2 Inter-Annotator Agreement

| Model | Agreement % | Cohen's κ | Interpretation |
|-------|-------------|-----------|----------------|
| T5-Joint | 92.1% | 0.839 | Excellent |
| T5-Control | 94.3% | 0.883 | Excellent |
| BART-RL | 94.3% | 0.884 | Excellent |

κ > 0.8 indicates excellent agreement. Human evaluation is reliable ground truth.

### 4.3 Cohen's Kappa Interpretation

| κ Value | Interpretation |
|---------|----------------|
| 1.0 | Perfect agreement |
| 0.8 - 1.0 | Almost perfect (excellent) |
| 0.6 - 0.8 | Substantial (good) |
| 0.4 - 0.6 | Moderate |
| 0.2 - 0.4 | Fair |
| 0.0 | No better than random |
| < 0 | Worse than random (anti-correlated) |

### 4.4 Key Finding: Classifier vs Human Gap

| Model | Classifier Flip | Human Flip (Strict) | Human Flip (Lenient) | Classifier-Human κ |
|-------|-----------------|---------------------|----------------------|--------------------|
| T5-Joint | 24.3% | 54.3% | 62.1% | -0.067 |
| T5-Control | 21.4% | 54.3% | 60.0% | -0.144 |
| BART-RL | 18.6% | 52.9% | 58.6% | -0.186 |

**Negative κ means the classifier is anti-correlated with human judgment.** The classifier is systematically fooled by surface edits (lowercase, punctuation removal) while missing actual sarcasm removal.

---

## 5. Model Comparison Summary

### 5.1 Large-Scale Evaluation (2857 samples)

| Model | Flip Rate | Similarity | Edit Dist | Perplexity | Behavior |
|-------|-----------|------------|-----------|------------|----------|
| T5-Joint | 21.4% | 0.87 | 0.60 | 635 | Conservative, preserves meaning |
| T5-Control | 21.0% | 0.88 | 0.59 | 614 | Conservative |
| T5 Ablations (6) | 20.5-21.3% | 0.88 | 0.57 | 590-607 | No significant difference |
| BART-base | 20.8% | 0.85 | 0.66 | 518 | Moderate rewriting |
| BART-CE | 21.5% | 0.64 | 0.92 | 364 | Aggressive, loses meaning |
| BART-CE-RL | 21.1% | 0.61 | 0.93 | 457 | Most aggressive |
| BART-RL | 21.2% | 0.85 | 0.61 | 726 | Moderate |
| LLaMA 1B | 21.9% | 0.66 | 0.95 | 378 | Complete rewrite |

**Observation:** All models get ~21% classifier flip rate despite vastly different behaviors (edit distance 0.57 to 0.95). This confirms the classifier limitation — it cannot distinguish conservative editing from aggressive rewriting.

### 5.2 Golden Data Evaluation (140 samples, human-annotated)

| Model | Human Flip (Strict) | Human Flip (Lenient) | Meaning Change | Strict Success |
|-------|---------------------|----------------------|----------------|----------------|
| **T5-Joint** | 54.3% | 62.1% | **16.4%** | **43.6%** |
| T5-Control | 54.3% | 60.0% | 25.0% | 39.3% |
| BART-RL | 52.9% | 58.6% | 40.7% | 34.3% |

**Key Finding:** Joint model has significantly lower meaning change than Control (16.4% vs 25.0%), leading to higher strict success (43.6% vs 39.3%). This difference is invisible to the automated classifier but revealed by human evaluation.

**Why Joint beats Control:**  
The strategy prefix forces the model to decompose the task:
1. First IDENTIFY what type of sarcasm (irony, satire, rhetorical question, etc.)
2. Then DECIDE what to preserve vs remove

Control learns blind input→output mapping without understanding the sarcasm structure.

---

## 6. Subtype Analysis

### 6.1 Classifier Flip Rate by Subtype (Golden Data, 140 samples)

| Subtype | Count | T5-Joint | T5-Control | BART-RL |
|---------|-------|----------|------------|---------|
| rhetorical_question | 14 | 50.0% | 21.4% | 21.4% |
| understatement | 11 | 36.4% | 27.3% | 9.1% |
| overstatement | 7 | 28.6% | 28.6% | 28.6% |
| irony | 30 | 23.3% | 20.0% | 16.7% |
| sarcasm | 69 | 20.3% | 21.7% | 20.3% |
| satire | 9 | 0.0% | 11.1% | 11.1% |

**Why rhetorical questions are easiest:**  
They have explicit structural markers ("What could go wrong?", "Isn't that great?") that models can learn to remove.

**Why satire is hardest:**  
Satire requires world knowledge and cultural context that small models don't have.

---

## 7. Running the Pipeline

### 7.1 Command

```bash
python scripts/eval_pipeline.py \
    --input model_outputs_clean/t5_base_joint.csv \
    --output results/t5_base_joint_results.csv \
    --gemini_key $GEMINI_API_KEY
```

### 7.2 Options

| Flag | Description |
|------|-------------|
| `--input` | Path to input CSV |
| `--output` | Path to output results CSV |
| `--skip_judge` | Skip LLM-as-Judge (faster, no API cost) |
| `--gemini_key` | API key for Gemini (required for LLM judge) |

### 7.3 Input Format

CSV with columns: `id, input, output, subtype`

### 7.4 Output Format

CSV with all computed metrics per sample:
- `id, input, output, subtype`
- `input_sarc_prob, output_sarc_prob, hard_flipped, flip_delta`
- `similarity`
- `perplexity`
- `bleu`
- `edit_dist_raw, edit_dist_norm`
- `paraphrase_score`
- `llm_sarcasm_removed, llm_meaning_preserved, llm_fluency` (if judge enabled)

---

## 8. File Structure

```
Project-LLMao/
├── data/
│   ├── golden/
│   │   ├── raw/                    # Human-annotated CSVs from Google Sheets
│   │   └── cleaned/                # Standardized format for pipeline
│   └── splits/                     # Train/val/test splits
├── model_outputs_clean/            # Model outputs (14 models)
├── results/
│   ├── golden/                     # Golden data evaluation results
│   │   ├── t5_base_joint_results.csv
│   │   ├── t5_base_control_results.csv
│   │   ├── bart_base_rl_results.csv
│   │   ├── *_merged.csv            # Human + automated merged
│   │   └── summary.csv             # Comparison table
│   └── *.csv                       # Full dataset results (2857 samples)
├── scripts/
│   ├── eval_pipeline.py            # Main evaluation script
│   ├── clean_golden_data.py        # Standardize human eval CSVs
│   └── analyze_golden_results.py   # Merge and analyze results
└── docs/
    └── EVALUATION.md               # This file
```

---

## 9. Conclusion

### What Each Metric Captures and Misses

| Metric | Captures | Misses |
|--------|----------|--------|
| Flip Rate | Binary sarcasm detection | Surface edits fool it |
| Similarity | Meaning preservation | Doesn't verify correctness |
| Perplexity | Fluency | Fluent nonsense passes |
| BLEU vs Input | Copying behavior | Doesn't measure quality |
| Edit Distance | Amount of change | Doesn't measure direction |
| LLM Judge | Holistic quality | Only 50 samples, may have bias |
| Human Eval | Ground truth | Expensive, limited scale |

### Key Takeaways

1. **Automated classifier is unreliable** — negative correlation with human judgment (κ = -0.07 to -0.19)
2. **Joint model beats Control** — 16.4% vs 25.0% meaning change, invisible to classifier
3. **Multiple metrics needed** — no single metric captures sarcasm removal + meaning preservation + fluency
4. **Human evaluation is essential** — the gap between machine and human evaluation is itself a finding

### The Evaluation Story

> "Automated metrics are necessary but not sufficient. The gap between machine and human evaluation reveals fundamental limitations in how we evaluate style transfer systems. Our 7-metric pipeline provides complementary views, but human judgment remains the gold standard for sarcasm rewriting evaluation."

---

## 10. References

- Cardiff NLP Twitter RoBERTa Irony: https://huggingface.co/cardiffnlp/twitter-roberta-base-irony
- Sentence Transformers: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Cohen's Kappa: Cohen, J. (1960). A coefficient of agreement for nominal scales.
- BLEU Score: Papineni et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation.
