# Experiments

> Per-model results and ablation findings.

The numbers here come from the actual eval CSVs in `results/`. The
[EVALUATION.md](EVALUATION.md) page has the full methodology and the
narrative analysis; the webapp `/dashboard` and `/human-eval` pages let
you slice the data interactively. This file is the static snapshot.

## Evaluation Scale

- **Full automated**: 2,857 samples × 14 models × 3 classifiers × 7 metrics
- **Human evaluation**: 140 samples × 3 models × 2 annotators (κ > 0.8)

## Headline Results

### Best model overall: T5-Joint

| Metric | T5-Joint | T5-Control | BART-RL | Notes |
|---|---|---|---|---|
| **Strict success (human)** | **43.6%** | 39.3% | 34.3% | Sarcasm removed AND meaning preserved |
| Human flip rate | 54.3% | 54.3% | 52.9% | All three remove sarcasm at similar rates |
| Meaning change rate | **16.4%** | 25.0% | 40.7% | T5-Joint preserves meaning best |
| Inter-annotator κ | 0.839 | 0.883 | 0.884 | Annotation is reliable |

T5-Joint wins because the strategy-prediction prefix forces the model to
identify *what's sarcastic* before rewriting. It's the same recipe and
data as T5-Control — the only difference is the joint task formulation.

## Full 7-Metric Summary (All 14 Models)

| Model | Flip Rate† | Similarity | Perplexity | BLEU | Edit Dist | Para Score |
|---|---|---|---|---|---|---|
| **t5_base_joint** | 8.4% | **0.870** | 571 | 0.188 | 0.630 | 0.170 |
| **t5_control** | 8.1% | **0.878** | 614 | 0.216 | 0.592 | 0.197 |
| joint (T5-small) | 8.4% | 0.872 | 635 | 0.211 | 0.600 | 0.192 |
| **bart_base_rl** | 8.8% | 0.852 | 726 | 0.216 | 0.606 | 0.198 |
| bart_base | 8.4% | 0.853 | **518** | 0.160 | 0.661 | 0.143 |
| bart_base_ce | 13.1% | 0.636 | 364 | 0.023 | 0.923 | 0.018 |
| bart_base_ce_rl | 12.6% | 0.609 | 457 | 0.021 | 0.928 | 0.015 |
| llama_3_2_1b | 13.7% | 0.656 | 378 | 0.013 | **0.948** | 0.009 |
| abl_without_irony | 8.0% | 0.882 | 591 | 0.234 | 0.575 | 0.213 |
| abl_without_overstatement | 8.4% | **0.885** | 607 | **0.235** | **0.570** | **0.215** |
| abl_without_rhet_q | 8.2% | 0.881 | 599 | 0.234 | 0.573 | 0.214 |
| abl_without_sarcasm | 7.7% | 0.883 | 608 | **0.237** | 0.568 | **0.217** |
| abl_without_satire | 8.0% | 0.880 | 594 | 0.229 | 0.579 | 0.209 |
| abl_without_understatement | 8.2% | 0.881 | 590 | 0.226 | 0.580 | 0.207 |

†Flip rate shown for RoBERTa-Twitter only — see "Multi-classifier audit"
below for the spread across all three classifiers.

## Multi-classifier Audit

Every test output was scored by three different sarcasm classifiers
trained on different domains. They disagree by up to 33 percentage points
on the same outputs.

| Model | RoBERTa-Twitter | Bert-Kaggle | RoBERTa-News | Spread |
|---|---|---|---|---|
| t5_base_joint | 8.4% | 41.6% | 21.2% | 33.2 pp |
| t5_control | 8.1% | 40.9% | 20.3% | 32.9 pp |
| joint | 8.4% | 41.1% | 22.4% | 32.7 pp |
| bart_base_rl | 8.8% | 39.9% | 26.4% | 31.1 pp |
| bart_base | 8.4% | 41.8% | 21.7% | 33.4 pp |
| bart_base_ce | 13.1% | 17.9% | 3.4% | 14.5 pp |
| bart_base_ce_rl | 12.6% | 39.6% | 8.2% | 31.4 pp |
| llama_3_2_1b | 13.7% | 16.3% | 4.8% | 11.5 pp |
| **AVERAGE** | **9.3%** | **37.5%** | **17.3%** | **28.2 pp** |

12 of 14 models show >30pp spread. **Pick whichever classifier supports
your hypothesis.** This is exactly why the human-eval audit was
necessary.

## Classifier vs Human Ground Truth

The 9-cell breakdown (3 models × 3 classifiers, on the 140-sample golden
data):

| Classifier | Avg Accuracy | Avg κ vs human |
|---|---|---|
| RoBERTa-Twitter | 47.6% | +0.019 |
| Bert-Kaggle | 43.1% | **−0.075** |
| RoBERTa-News | 53.6% | +0.104 |
| **Human inter-annotator** | — | **0.836–0.884** |

**4 of 9 model×classifier cells show negative κ** — the classifier
anti-correlates with human judgment. The best κ across the entire matrix
is +0.18 (RoBERTa-News on T5-Joint) compared to 0.84+ for human
agreement.

**Takeaway: automated flip rate is not a valid primary metric.**
Classifiers detect sarcasm presence, not removal between input and
output.

## Per-Subtype Performance (T5-Joint, full test set)

| Subtype | N | Flip Rate | Similarity | BLEU | Para Score |
|---|---|---|---|---|---|
| sarcasm | 883 | 7.5% | 0.867 | 0.213 | 0.193 |
| irony | 614 | 9.6% | 0.874 | 0.212 | 0.193 |
| satire | 525 | 6.1% | 0.888 | 0.222 | 0.204 |
| overstatement | 401 | 8.7% | 0.873 | 0.224 | 0.204 |
| understatement | 330 | 9.7% | 0.862 | 0.183 | 0.164 |
| rhetorical_question | 104 | 13.5% | 0.840 | 0.169 | 0.148 |

Rhetorical questions are the hardest by every metric — the sarcasm lives
in the implication, not the lexical surface, and the model has the
fewest training examples for them.

## Human Evaluation by Subtype (T5-Joint, golden data)

| Subtype | N | Human Flip | Meaning Δ | Strict Success |
|---|---|---|---|---|
| sarcasm | 69 | 59.4% | 23.2% | 44.9% |
| irony | 30 | 50.0% | 6.7% | 50.0% |
| rhetorical_question | 14 | 50.0% | 0.0% | 50.0% |
| understatement | 11 | 54.5% | 18.2% | 36.4% |
| satire | 9 | 55.6% | 33.3% | 22.2% |
| overstatement | 7 | 28.6% | 0.0% | 28.6% |

## Ablation Study (Subtype Held-out)

Six retrains of the T5 control recipe with one subtype dropped from
train+val. Pools are stratified-downsampled to the minimum across all six
to keep effective dataset size constant; test set is the full split.

| Held out | Similarity | BLEU | Para Score |
|---|---|---|---|
| (T5-Control baseline) | 0.878 | 0.216 | 0.197 |
| sarcasm | 0.883 | 0.237 | 0.217 |
| irony | 0.882 | 0.234 | 0.213 |
| satire | 0.880 | 0.229 | 0.209 |
| overstatement | 0.885 | 0.235 | 0.215 |
| understatement | 0.881 | 0.226 | 0.207 |
| rhetorical_question | 0.881 | 0.234 | 0.214 |

**All six ablations cluster within 0.005 similarity** of each other.

**Interpretation**: sarcasm subtypes share underlying mechanisms
(hyperbole, contradiction, absurdity) and the model learns generic
patterns that transfer across categories. No single subtype is
load-bearing — a positive finding for generalisation, and an
explanation for why the model can't be trivially improved by
oversampling one subtype.

## Key Findings

1. **Automated flip rate is unreliable.** All three classifiers fail
   against human ground truth (κ −0.11 to +0.18 vs human κ > 0.8).
   Use multi-classifier + human eval, never a single flip rate alone.

2. **T5-Joint is the best model.** Strict success rate of 43.6%, beating
   T5-Control (39.3%) and BART-RL (34.3%). The strategy prefix forces
   task decomposition; same data, same recipe — only the task
   formulation changes.

3. **BART-RL is reward hacking.** Highest classifier flip rates but
   40.7% meaning-change rate on human eval. The composite reward is
   satisfied by deleting sarcastic tokens; KL penalty isn't enough to
   stop it.

4. **LLaMA and BART-CE rewrite too aggressively.** Edit distance ~0.95,
   BLEU ~0.01 — these are *generations*, not transfers. Similarity
   drops to 0.61–0.66 and human eval shows the meaning is gone.

5. **Subtype ablations are interchangeable.** Six retrains all within
   0.005 similarity. Dropping any one subtype doesn't cripple the model.

6. **Different subtypes fail for different reasons.** Satire mimics
   real news (highest classifier miss rate at 80%); rhetorical questions
   encode sarcasm in implication, not lexicon; overstatement is a
   *model* failure (only 28.6% human flip rate — the model can't remove
   what defines the headline).

## Reproducing the Results

```bash
# Yang Zhi's BART + LLaMA pipeline
python scripts/train.py --model facebook/bart-base --direction sar-to-non
python scripts/train_rl.py --sft_checkpoint outputs/bart-base/sar-to-non/final
python scripts/train_llama.py
python scripts/train_llama_context.py

# Camille's T5 pipeline (separate repo)
# https://github.com/camille-readbean/CS4248-project-AY2526S2
sbatch scripts/slurm_finetune_t5.sh --model t5-base
sbatch scripts/slurm_finetune_t5_control.sh t5-base

# Evaluation (this repo)
python scripts/batch_eval.py --multi_classifier
python scripts/analyze_golden_results.py
```

Output files land in `results/` and `results/golden/`. The webapp's
static export script (`webapp/backend/scripts/export_static.py`) reads
these and produces the JSON consumed by the Next.js dashboard.

---

_Last updated: 2026-04-14_
