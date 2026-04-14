# Poster Content: Project LLMao

> **LLMao: Lightweight Language Models for Anti-sarcasm Output**

**CS4248 Team 14**: 
**Mentor**: Xiao Yu

---

## Abstract

Sarcasm poses a significant challenge for NLP systems — sentiment
analyzers misread sarcastic text as positive, and content moderation
pipelines fail to capture the intended meaning. We tackle **sarcasm
style transfer**: given a sarcastic headline, generate a non-sarcastic
equivalent that preserves the underlying meaning. Since no large-scale
paired dataset exists for this task, we construct a synthetic parallel
corpus of 89,688 strategy-annotated pairs using LLM-based generation
(StepFun Step-3.5 Flash) with cross-validation (Nemotron 3 Nano). We
train **14 models** across four recipes — supervised seq2seq fine-tuning
(T5 and BART variants), REINFORCE with KL penalty, LoRA instruction
tuning of LLaMA 3.2 1B, and a 6-way subtype ablation — and evaluate
them with a 7-metric automated pipeline plus 3 sarcasm classifiers and
a hand-labeled human evaluation on 140 samples (κ > 0.8). Our headline
finding: **all three classifiers fail vs human ground truth**
(κ −0.11 to +0.18), making automated flip rate an unreliable primary
metric. Our **best model is T5-Joint** — a T5-base trained to predict
the sarcasm strategy *before* rewriting — which achieves 43.6% strict
success (sarcasm removed AND meaning preserved), beating both T5-Control
and BART-RL on the same hand-labeled set.

---

## 1. Data Pipeline

### 1.1 Source Dataset

**News Headlines Dataset for Sarcasm Detection (NHDSD)**
- 28,619 headlines: 13,634 sarcastic (TheOnion) + 14,985 non-sarcastic (HuffPost)
- Professional writing, self-contained, high-quality labels

### 1.2 Data Cleaning & Reclassification

```
NHDSD (28,619) → Clean & dedup (28,497) → LLM reclassify → Cross-validate disagreements
```

1. **Binary reclassification** with StepFun Step-3.5 Flash — agreement with original: 80.19%
2. **Cross-validation** of 5,644 disagreements with Nemotron — 72.2% confirmed as NHDSD mislabels
3. **Censorship filtering**: 9 headlines removed (content filter blocks)

### 1.3 Synthetic Parallel Corpus Construction

No paired sarcasm↔non-sarcasm dataset exists at scale. We use LLM generation to create one:

| Direction | Pairs Generated |
|-----------|----------------|
| Sarcastic → Non-sarcastic | 13,588 |
| Non-sarcastic → Sarcastic | 14,948 |
| **Total** | **28,536** |

Each pair is annotated with one of **6 sarcasm strategies** (from iSarcasmEval taxonomy):

| Strategy | Description |
|----------|-------------|
| `sarcasm` | Contradicts state of affairs, critical |
| `irony` | Contradicts state of affairs, not critical |
| `satire` | Appears supportive, contains mockery |
| `understatement` | Undermines importance |
| `overstatement` | Obviously exaggerated |
| `rhetorical_question` | Question contradicting reality |

### 1.4 Strategy Augmentation

For each non→sarcastic pair, generate 5 additional strategy variants → **89,688 total records** with perfect strategy balance (14,948 per strategy).

### 1.5 Train/Val/Test Splits

**Non-to-sarcastic** (strategy-augmented, source-level split to prevent leakage):

| Split | Sources | Records |
|-------|---------|---------|
| Train | 11,955 | 71,730 |
| Val | 1,492 | 8,952 |
| Test | 1,501 | 9,006 |

**Sarcastic-to-non** (primary task, 80/10/10):

| Split | Records |
|-------|---------|
| Train | 10,868 |
| Val | 1,356 |
| Test | 1,364 |

---

## 2. Method

### Why LLMs for Data, Small Models for the Task

The LLM serves as a **synthetic data annotator** — creating paired training data that doesn't exist in the wild. The research question is whether small, efficient models can learn style transfer from this supervision.

| | LLM (data generation) | Small model (task) |
|---|---|---|
| **Role** | Annotator | Learner |
| **Inspectable** | Black box | Full ablation possible |
| **Inference** | ~1-2s, per-token cost | ~10ms on single GPU |
| **Control** | Prompt-dependent | Deterministic strategy codes |

### Model Architectures (14 models, four recipes)

**T5-Joint / T5-Control / 6 ablations** — `t5-base` (220M), trained in
[Camille's separate repo](https://github.com/camille-readbean/CS4248-project-AY2526S2)
via `finetune_T5.py` with SLURM orchestration. T5-Joint is conditioned
to predict the sarcasm strategy in its target string (`strategy: X
rewrite: Y`); T5-Control uses a plain `rewrite to non-sarcastic:`
prefix and emits the bare rewrite. The six ablation models drop one
subtype each from train+val. Same recipe across all nine: 4 epochs,
per-device batch 8 × grad accum 2, LR 3e-4, cosine warmup 0.06, max
sequence length 1248, fp16, eval_loss as best metric.

**BART-Base / BART-CE** — `facebook/bart-base` (140M), trained via
this repo's `scripts/train.py`. BART-Base trains on the main
`sar_to_non` split (10,868 pairs); BART-CE trains on the smaller
`sar_to_non_context_enhanced` split (8,258 pairs with scraped article
bodies). HuggingFace `Seq2SeqTrainer`, 5 epochs with early stop on val
BLEU (patience 2), batch 16, LR 3e-4, max length 128, bf16.

**BART-RL / BART-CE+RL** — `scripts/train_rl.py`. Initialise both
policy and frozen reference from the corresponding SFT checkpoint;
update the policy with REINFORCE + a KL penalty using a composite
classifier + ROUGE-L reward. See § "Reinforcement Learning" below for
the loss formulation.

**LLaMA 3.2 1B / LLaMA 3.2 1B (context)** —
`meta-llama/Llama-3.2-1B-Instruct` (1.24B), `scripts/train_llama.py`
and `train_llama_context.py`. PEFT LoRA (r=16, α=32, dropout 0.05) on
all 7 attention + MLP projections. ~6M of 1.24B params trainable
(0.5%). LR 2e-4, effective batch 16 (8 × 2 grad accum), 3 epochs,
cosine schedule, 5% warmup, bf16 + gradient checkpointing. Loss is
masked to the assistant response only. The context variant adds the
scraped article body to the user message at max length 1024.

**Llama-3.2-1B-Instruct LoRA Configuration:**

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 2e-4 |
| Batch size (effective) | 16 (8 × 2 grad accum) |
| Max epochs | 3 (best at epoch 1) |
| Max sequence length | 256 |
| LoRA rank (r) | 16 |
| LoRA alpha (α) | 32 |
| LoRA dropout | 0.05 |
| LR scheduler | cosine |
| Warmup | 5% of steps |
| Gradient checkpointing | enabled |
| Precision | bfloat16 |

### Stage 2: Reinforcement Learning (BART-base)

SFT alone produces surface paraphrasing. Following ViSP (2025), we apply **REINFORCE with KL penalty** to refine the SFT-trained BART using our sarcasm classifier as the reward signal.

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────┐
│ SFT BART    │ ──► │ Generate via  │ ──► │ Sarcasm        │ ──► │ REINFORCE│
│ (policy π)  │     │ sampling     │     │ Classifier     │     │ update   │
└─────────────┘     └──────────────┘     │ (reward model) │     └──────────┘
                                          │ r = 1-P(sarc)  │         │
       ┌─────────────┐                    └────────────────┘         │
       │ Frozen SFT   │ ─── KL(π_RL || π_SFT) penalty ──────────────┘
       │ (reference)  │
       └─────────────┘
```

**Loss function**: `L = L_REINFORCE + β · KL(π_RL || π_SFT)`

**Reward**: Composite score —
`r = α · (1 - P(sarcastic)) + (1-α) · ROUGE-L(output, reference)`

- Style reward from a held-out sarcasm classifier (high reward =
  output reads as non-sarcastic). Pure style reward saturates because
  the SFT outputs already classify as ~1.0 non-sarcastic.
- Content reward from ROUGE-L penalises outputs that lose meaning.

**KL penalty**: in theory prevents reward hacking by keeping the policy
close to the coherent SFT reference. In practice, β = 0.2 is *not*
enough — see Finding 3 in the analysis section: BART-RL achieves the
highest classifier flip rates by deleting sarcastic tokens, and human
eval shows it has a 40.7% meaning-change rate.

| RL Hyperparameter | Value |
|-------------------|-------|
| Learning rate | 1e-5 |
| Batch size | 8 |
| Epochs | 3 (best at epoch 2) |
| KL coefficient (β) | 0.2 |
| Style weight (α) | 0.5 |
| Reward | α · (1 - P(sarcastic)) + (1-α) · ROUGE-L |
| Sampling | top-k=50, top-p=0.95, temp=0.8 |
| Reward baseline | EMA (momentum=0.9) |
| Gradient clipping | max_norm=1.0 |

---

## 3. Results

### 3.1 Automatic Metrics (2,857-sample Test Set)

| Model | Similarity ↑ | BLEU vs input | Edit Dist | Para Score ↑ | RoBERTa-Twitter Flip |
|-------|---|---|---|---|---|
| **T5-Joint** | **0.870** | 0.188 | 0.630 | 0.170 | 8.4% |
| T5-Control | 0.878 | 0.216 | 0.592 | 0.197 | 8.1% |
| BART-Base | 0.853 | 0.160 | 0.661 | 0.143 | 8.4% |
| BART-RL | 0.852 | 0.216 | 0.606 | 0.198 | 8.8% |
| BART-CE | 0.636 | 0.023 | 0.923 | 0.018 | 13.1% |
| BART-CE+RL | 0.609 | 0.021 | 0.928 | 0.015 | 12.6% |
| LLaMA 3.2 1B | 0.656 | 0.013 | **0.948** | 0.009 | 13.7% |

*Paraphrase score = similarity × (1 − BLEU vs input); higher = genuine
rewriting that preserves meaning.*

### 3.2 Multi-Classifier Audit — Same Outputs, Three Stories

| Model | RoBERTa-Twitter | Bert-Kaggle | RoBERTa-News | Spread |
|---|---|---|---|---|
| T5-Joint | 8.4% | 41.6% | 21.2% | 33.2 pp |
| BART-RL | 8.8% | 39.9% | 26.4% | 31.1 pp |
| LLaMA 3.2 1B | 13.7% | 16.3% | 4.8% | 11.5 pp |

12 of 14 models show >30 pp spread. Pick whichever classifier supports
your hypothesis.

### 3.3 Human Evaluation (140 Samples × 3 Models × 2 Annotators)

| Metric | T5-Joint | T5-Control | BART-RL |
|---|---|---|---|
| Inter-annotator κ | 0.839 | 0.883 | 0.884 |
| Human flip rate | 54.3% | 54.3% | 52.9% |
| Meaning change rate | **16.4%** | 25.0% | 40.7% |
| **Strict success** | **43.6%** | 39.3% | 34.3% |

Strict success = sarcasm removed AND meaning preserved.

### 3.4 Classifier vs Human (Cohen's κ)

| Classifier | Avg κ vs Human |
|---|---|
| RoBERTa-Twitter | +0.019 |
| Bert-Kaggle | **−0.075** |
| RoBERTa-News | +0.104 |
| **Human inter-annotator** | **0.84+** |

4 of 9 model×classifier cells show negative κ.

### 3.2 Sample Outputs — Onion Headlines (SFT vs RL)

| Input (sarcastic) | BART (SFT) | BART (SFT + RL) | Llama 3.2 1B (SFT LoRA) |
|--------------------|------------|------------------|--------------------------|
| "Panicked Dodgers Owner Has No Idea How He's Going To Come Up With $414 Million Payroll" | dodgers owner unsure how to raise $414 million | The panicked Dodgers owner is unsure how he will come up with $414 million payroll. | Dodgers Owner Aims To Sign Top Free Agent To $414 Million Contract |
| "Hospital Decides Cancer-Sniffing Leopard More Trouble Than It's Worth" | hospital warns cancer-sniffing leopard may cause more trouble than it's worth | Hospital believes a cancer-sniffing leopard is causing more trouble. | Leopard Found in Hospital Waiting Room |
| "Jake Paul Hoping To Gain Respect Of Boxing Community With Fight Against World's Tallest Man" | jake paul hopes to gain respect of the boxing community with fight against world's tallest man | Jake Paul hopes to gain respect of the Boxing Community with a fight against the world's tallest man. | Jake Paul Prepares For Upcoming Boxing Match Against World's Tallest Man |
| "Nation Likes When Bib Has Picture Of Food They Eating" | the nation loves when a bib has a picture of food they eat | People love when Bib has a picture of food they eat. | Biblically-Inspired Food-Themed Bibs Gain Popularity |
| "Mail Carrier Hurt To Learn Residents On His Route Have Been Receiving Electronic Mail" | Mail carrier hurt to learn residents on his route have been receiving electronic mail | Mail carrier was hurt to learn that residents on his Route have been receiving electronic mail. | Mail Carrier Reports Residents On His Route Have Been Receiving Electronic Mail |

**Failure modes (RL):**
- Empty outputs for very short/vulgar inputs (e.g., "Fucker Has Nerve To Be 22 Years Old" → empty string) — reward hacking via trivially non-sarcastic empty text
- Verbatim copy for ultra-short inputs (e.g., "Norris God" → "norris god")

**Failure modes (Llama SFT):**
- **Hallucination**: Model invents plausible but incorrect facts (e.g., "Fucker Has Nerve To Be 22 Years Old" → "Local Man Arrested for Sexual Assault at 22" — fabricated crime)
- **Meta-descriptions**: Describes the article instead of rewriting the headline (e.g., "Norris God" → "Satirical Article Features Fictional God Named Norris")
- **Meaning drift**: Rewrites lose the core meaning while producing a valid headline (e.g., "Nation Likes When Bib Has Picture Of Food They Eating" → "Biblically-Inspired Food-Themed Bibs Gain Popularity")

### 3.3 Strategy Breakdown (BLEU)

<!-- TODO: Fill from eval_results.json -->

| Strategy | T5 | BART | GPT-2 |
|----------|----|------|-------|
| sarcasm | | | |
| irony | | | |
| satire | | | |
| understatement | | | |
| overstatement | | | |
| rhetorical_question | | | |

---

## 4. Analysis & Key Findings

### Finding 1: Automated Flip Rate is Not a Valid Primary Metric

Three sarcasm classifiers from different domains (Twitter, Kaggle
headlines, news headlines) disagree by up to 33 percentage points on
the same outputs and all three score Cohen's κ between −0.11 and +0.18
against human ground truth (where annotators agree at κ > 0.8). 4 of 9
model×classifier cells show *negative* κ — the classifier
anti-correlates with humans. The classifiers detect sarcasm presence
in isolation, not removal between input and output. **Any single
flip-rate number is meaningless without the spread.**

### Finding 2: T5-Joint is the Best Model Overall

T5-Joint achieves the highest strict-success rate on human evaluation
(43.6% — sarcasm removed AND meaning preserved), beating both T5-Control
(39.3%) and BART-RL (34.3%). The structural difference is just the
target format:

- T5-Joint target: `"strategy: {strategy} rewrite: {non_sarcastic}"`
- T5-Control target: `{non_sarcastic}` (plain)

Forcing the model to predict the strategy *before* generating the
rewrite makes it decompose the task. Same data, same recipe — only the
joint task formulation changes — and meaning preservation improves from
75% (control) to 84% (joint). This is the central positive finding of
the project.

### Finding 3: BART-RL is Reward Hacking

BART-RL achieves the highest classifier flip rates of any model but
human evaluation reveals a **40.7% meaning change rate** — more than
double T5-Joint's 16.4%. The composite reward
(`α · (1−P(sarc)) + (1−α) · ROUGE-L`) is satisfied by deleting
sarcastic tokens while keeping enough overlap to satisfy ROUGE-L.
Deletion ≠ rewriting. The KL penalty against the SFT reference is not
strong enough to prevent it.

```
Input:  "Area Man Proud Of Completely Average Achievement"
BART-RL: "Man achieves something."
```

The classifier is happy. The human annotator marks the meaning as
destroyed. **This is the cautionary finding of the project**: optimising
a reward you can't trust produces a model you can't trust.

### Finding 4: LLaMA and BART-CE Rewrite Too Aggressively

| Model | Similarity | Edit Distance | LLM Meaning |
|---|---|---|---|
| LLaMA 3.2 1B | 0.66 | 0.95 | 3.34 / 5 |
| BART-CE | 0.64 | 0.92 | 3.52 / 5 |
| BART-CE+RL | 0.61 | 0.93 | 2.90 / 5 |

These models edit ~95% of the input and retain ~1% n-gram overlap.
That's *generation* with the input as a prompt, not *style transfer*.
The hallucination footprint is real:

```
Input:  "Fucker Has Nerve To Be 22 Years Old"
LLaMA:  "Local Man Arrested for Sexual Assault at 22"   (fabricated crime)

Input:  "Inconsiderate Wife Leaves Bathroom A Total Mess After Home Birth"
LLaMA:  "Mother of Two Gives Birth at Home"             (loses the absurdity)
```

LLaMA's larger backbone enables genuine comprehension but it has too
much generative latitude — it forgets to preserve the original.

### Finding 5: Subtype Ablations Are Interchangeable

Six retrains of the T5 control recipe, each with one of the six sarcasm
subtypes dropped from train+val (pools downsampled to keep effective
dataset size constant). All six cluster within **0.005 similarity** of
each other. No single subtype is load-bearing.

**Interpretation**: sarcasm subtypes share underlying mechanisms
(hyperbole, contradiction, absurdity) and the model learns generic
patterns that transfer across categories. A positive generalisation
finding, and an explanation for why oversampling one subtype doesn't
help.

### Finding 6: Different Subtypes Fail for Different Reasons

| Subtype | Classifier Miss Rate† | Why |
|---|---|---|
| satire | 80% | Mimics legitimate news format |
| rhetorical_question | 71% | Sarcasm in pragmatics, not lexicon |
| irony | 67% | Contradiction is contextual, no surface markers |
| understatement | 50% | Requires world knowledge of "appropriate" response |
| sarcasm (generic) | 44% | Has detectable lexical patterns |
| overstatement | 100%* | **Model failure**: only 28.6% human flip rate |

†% of human-labeled flips that the classifier failed to detect.
*Overstatement is anomalous: 100% miss rate isn't classifier failure
but model failure — only 28.6% of overstatement headlines are
successfully de-sarcasm'd by humans either.

### The Knowledge Gap

Sarcasm comprehension requires:
1. **World knowledge** — understanding what's normal vs. absurd
2. **Pragmatic inference** — recognizing speaker intent vs. literal meaning
3. **Cultural context** — knowing that TheOnion headlines follow specific
   comedic patterns

Small SFT models pattern-match on lexical surface; RL pushes them toward
optimising whatever the reward says, including hacks. The strategy-prefix
joint task (T5-Joint) is the only training-time intervention that
demonstrably improves *meaning preservation* on human eval, by forcing
the model to identify what's sarcastic before rewriting it.

---

## 5. Error Taxonomy

**BART (SFT) error patterns:**

| Error Type | Example | Frequency |
|------------|---------|-----------|
| **Capitalization-only** | "area man" → "Area Man" | High |
| **Article insertion** | "man says" → "A man says" | High |
| **Punctuation addition** | no period → added period | Medium |
| **Minor word substitution** | "passionate" → "devoted" | Low |
| **Actual de-sarcasm** | Meaningful rewrite | Rare |

**Llama-3.2-1B (SFT LoRA) error patterns:**

| Error Type | Example | Frequency |
|------------|---------|-----------|
| **Hallucination** | "Fucker Has Nerve To Be 22 Years Old" → "Local Man Arrested for Sexual Assault at 22" | Medium |
| **Meta-description** | "Norris God" → "Satirical Article Features Fictional God Named Norris" | Medium |
| **Meaning drift** | Headline loses original topic while remaining non-sarcastic | Low |
| **Classifier false negative** | Output reads as non-sarcastic but classifier disagrees | Low |
| **Genuine de-sarcasm** | "Inconsiderate Wife Leaves Bathroom A Total Mess After Home Birth" → "Mother of Two Gives Birth at Home" | High |

---

## 6. Conclusion

- We construct an **89,688-record strategy-annotated parallel corpus**
  via LLM annotation + cross-validation — a reusable resource and the
  source of training data for all 14 models.
- We train **14 models across four recipes**: BART SFT (Yang Zhi), T5
  joint/control/ablations (Camille's separate pipeline), REINFORCE + KL
  on BART, and LoRA instruction tuning of LLaMA 3.2 1B.
- **All three sarcasm classifiers fail against human ground truth**
  (Cohen's κ −0.11 to +0.18 vs human κ > 0.8). 4 of 9 model×classifier
  cells show negative κ. **Automated flip rate is not a valid primary
  metric** — classifiers detect sarcasm presence, not removal between
  input and output.
- **T5-Joint is our best model** (43.6% strict success on human eval),
  beating both T5-Control (39.3%) and BART-RL (34.3%). The strategy-
  prediction prefix forces task decomposition before generation, and
  this single change improves meaning preservation from 75% to 84% on
  the same data and recipe.
- **BART-RL is reward hacking.** It scores highest on automated flip
  rate but human eval shows 40.7% meaning change. The composite
  reward is satisfied by deleting sarcastic tokens; the KL penalty
  isn't enough to stop it. A structural cautionary finding about
  optimising rewards you can't trust.
- **The 6-way subtype ablation is null** — all six retrained models
  cluster within 0.005 similarity. Sarcasm subtypes share underlying
  mechanisms; no single one is load-bearing.
- **Different subtypes fail for different reasons**: satire mimics
  legitimate news; rhetorical questions encode sarcasm in pragmatics;
  overstatement is a model failure (only 28.6% human flip rate).
- **Limitation — domain specificity**: Sarcasm detection is strongly
  domain-dependent. The three off-the-shelf classifiers we audit were
  each trained on a different corpus (Twitter irony, Kaggle headlines,
  news headlines) and disagree with each other by up to 33 percentage
  points on the same outputs. The RL reward signal inherits this
  fragility — a setup that is only valid for the specific classifier
  chosen and the domain it was trained on.
- **Future work**: DPO as an alternative to REINFORCE; train a
  domain-general sarcasm classifier as a more reliable reward model;
  apply the joint-task prefix to BART and LLaMA backbones to test
  whether the gain is T5-specific; richer human eval with subtype-
  level rationales.

---

## References

- Misra, R. (2019). News Headlines Dataset for Sarcasm Detection. Kaggle.
- Abu Farha, I. et al. (2022). SemEval-2022 Task 6: iSarcasmEval.
- Raffel, C. et al. (2020). T5: Exploring the Limits of Transfer Learning.
- Lewis, M. et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training.
- Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners (GPT-2).
- Zhu, K. et al. (2025). ViSP: Visual Sarcasm Generation with PPO Reinforcement Learning.
- Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.
