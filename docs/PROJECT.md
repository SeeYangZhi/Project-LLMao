# Project LLMao

> **LLMao: Lightweight Language Models for Anti-sarcasm Output**

CS4248 · Team 14 · National University of Singapore · AY2025/26 S2

---

## Table of Contents

1. [TL;DR](#tldr)
2. [Background & Motivation](#1-background--motivation)
3. [Problem Formulation](#2-problem-formulation)
4. [Dataset Construction](#3-dataset-construction)
5. [Model Recipes](#4-model-recipes)
6. [Evaluation Methodology](#5-evaluation-methodology)
7. [Results](#6-results)
8. [Findings & Analysis](#7-findings--analysis)
9. [Limitations](#8-limitations)
10. [Future Work](#9-future-work)
11. [Webapp](#10-webapp)
12. [Reproducibility](#11-reproducibility)
13. [Lessons Learned](#12-lessons-learned)
14. [Glossary](#glossary)
15. [Appendix A — Full Hyperparameter Tables](#appendix-a--full-hyperparameter-tables)
16. [Appendix B — LLM Annotation Prompts](#appendix-b--llm-annotation-prompts)
17. [References](#references)

---

## TL;DR

We train and evaluate **14 small language models** (T5, BART, LLaMA — all
under 1.3B parameters) on **sarcasm style transfer**: given a sarcastic
news headline, generate a non-sarcastic equivalent that preserves the
underlying meaning. Three findings drive the project:

1. **Automated sarcasm classifiers cannot be trusted as a primary
   metric.** Three off-the-shelf classifiers from different domains
   disagree by up to 33 percentage points on the same outputs and all
   three score Cohen's κ between −0.11 and +0.18 against human ground
   truth (where annotators agree at κ > 0.8). Single-classifier flip
   rate is meaningless without the spread.
2. **The strategy-prefix joint task wins.** A T5-base model trained to
   predict the sarcasm strategy *before* rewriting (T5-Joint) achieves
   43.6% strict success on hand-labeled human evaluation, beating both
   T5-Control (39.3%) and BART-RL (34.3%) on the same 140-sample set.
   Same data, same recipe — only the joint task formulation changes —
   and meaning preservation improves from 75% to 84%.
3. **Reinforcement learning with classifier rewards is reward hacking.**
   BART-RL achieves the highest classifier flip rates of any model but
   human evaluation reveals a 40.7% meaning-change rate. The composite
   `style + ROUGE-L` reward is satisfied by deleting sarcastic tokens;
   the KL penalty against the SFT reference is not enough to stop it.

The full results, methodology, and an interactive dashboard live in the
[webapp](../webapp/) and the per-page documentation linked at the
bottom of this file.

---

## 1. Background & Motivation

### 1.1 The sarcasm problem in NLP

Sarcasm is a long-standing failure mode for natural language
understanding. Sentiment analysers misread *"Great job on the update,
everything is broken now"* as positive. Toxicity classifiers miss the
mockery in *"Oh sure, that's a totally reasonable take."* Content
moderation pipelines overlook satirical headlines that propagate
disinformation under the cover of irony. Survey responses, support
tickets, and social-media data — all of these contain sarcasm at rates
that materially affect downstream decision-making.

The dominant academic framing of the problem is **sarcasm detection**:
binary classification of "is this text sarcastic?" Decade-long efforts
have pushed news-headline classifiers above 95% accuracy on benchmarks,
but these models give a single bit of information. They can't tell you
*what* in the text reads as sarcastic, *which mechanism* the author
used (irony vs hyperbole vs rhetorical question), or *what the
underlying claim is*. They are also famously brittle out of domain — a
classifier trained on tweets struggles on news headlines and vice
versa, because the lexical surface differs.

### 1.2 Why style transfer instead of detection

We reframe the problem as **style transfer**: rather than detecting
sarcasm, *remove* it. Given a sarcastic headline, generate a neutral,
factual equivalent that preserves the underlying claim. This reframing
is more demanding in three ways:

1. **It forces comprehension.** You can't faithfully rewrite a headline
   you don't understand. Detection accuracy is lossy by design — it
   collapses everything to a single bit. Style transfer requires the
   model to recover the underlying claim and produce a grammatical,
   meaning-preserving rewrite.
2. **It produces a useful artefact.** The output is a sentiment-tagger-
   friendly version of the same news event that downstream pipelines
   (sentiment analysis, content moderation, summarisation, news
   aggregation) can ingest directly.
3. **It exposes the failure modes of pure metric optimisation.** When
   the only training signal you have is a sarcasm classifier's score
   on the output, the model will exploit that score in ways that don't
   reflect human judgment. The reward-hacking result we report below
   (Finding 7.3) is the natural consequence.

### 1.3 Why small models

We deliberately scope the work to **small language models** (60M–1.3B
parameters) rather than frontier LLMs. Three reasons:

| Property | Small model | Frontier LLM |
|---|---|---|
| **Inspectable** | Full ablation possible, deterministic outputs, can save and diff checkpoints | Black box; behaviour drifts across API versions |
| **Practical** | ~10ms inference on a single consumer GPU | 1–2s per call, per-token cost, network dependency |
| **Controllable** | Trained explicitly on strategy tokens for structured output | Prompt-dependent, no hard guarantee |
| **Reproducible** | Same seed, same code → same model | Provider-side updates can change behaviour overnight |

The frontier LLM still appears in the project — but as a **synthetic
data annotator**, not as the model that solves the task. This separates
the dataset construction question (how do we get supervision that
doesn't exist in the wild) from the research question (can small,
efficient models learn from that supervision). It also lets us run
ablations on the small model that would be infeasible on a 70B+ model
with a closed API.

### 1.4 Why a multi-recipe study

Most sarcasm-style-transfer papers train one model with one recipe and
report one number. We deliberately train **14 models across four
recipes** (BART SFT, T5 SFT, REINFORCE+KL, LoRA instruction tuning) so
that we can:

- Compare architectures (encoder-decoder vs decoder-only) on the same data
- Compare loss formulations (cross-entropy vs RL with classifier reward)
  on the same backbone
- Run a 6-way subtype ablation to test whether any sarcasm category is
  load-bearing
- Cross-validate findings across model families instead of overfitting
  to one architecture

The cost is real (≈100 GPU-hours total across the team) but the payoff
is that every claim in the Findings section is supported by multiple
models, not extrapolated from one.

---

## 2. Problem Formulation

### 2.1 Task definition

Let `H_sarc` be the space of sarcastic news headlines and `H_neutral`
the space of neutral, factual news headlines. Sarcasm style transfer
is the task of learning a function

```
f : H_sarc → H_neutral
```

such that for any input `x ∈ H_sarc`:

- `f(x)` is **non-sarcastic** (the sarcastic mechanism is removed)
- `f(x)` **preserves the underlying claim** of `x` (no meaning loss)
- `f(x)` is **fluent and grammatical**

Three conditions, all of which are necessary. A model that achieves
non-sarcasm by deleting tokens (Finding 7.3) violates the second.
A model that achieves meaning preservation by copying the input
(Finding 7.1, low BLEU vs input + high similarity heuristic) violates
the first. A model that achieves both via fabricated rewrites
(Finding 7.4) violates faithfulness in a different way.

### 2.2 The six sarcasm subtypes

We adopt the iSarcasm taxonomy (Abu Farha et al. 2022), which
distinguishes six mechanisms by which a text becomes sarcastic:

| Subtype | Mechanism | Example |
|---|---|---|
| **sarcasm** (generic) | Contradicts state of affairs, critical | *"Great job on the update, everything is broken now"* |
| **irony** | Contradicts state of affairs, not critical | *"I love waiting in line at the DMV"* |
| **satire** | Mimics a serious genre (here, news) with mockery | *"Congress Votes To Continue Doing Nothing"* |
| **overstatement** | Obviously exaggerated | *"Area Man Has Most Important Day Of His Life"* |
| **understatement** | Neutral words for extreme situations | *"Plane Crash Causes Some Concern"* |
| **rhetorical_question** | Question whose answer is implicit | *"Who Actually Believes This Works?"* |

These subtypes matter for two reasons. First, they're the basis of the
**joint task** that produces our best model: T5-Joint is trained to
emit `strategy: <subtype> rewrite: <output>` rather than just the
output, forcing it to identify the mechanism before rewriting. Second,
they're the basis of the **6-way ablation study**: each ablation drops
one subtype from training and measures the impact, letting us test
whether any subtype is load-bearing.

Subtype distribution in our test set (2,857 samples) is heavy-tailed:

| Subtype | Count | Share |
|---|---|---|
| sarcasm | 883 | 30.9% |
| irony | 614 | 21.5% |
| satire | 525 | 18.4% |
| overstatement | 401 | 14.0% |
| understatement | 330 | 11.6% |
| rhetorical_question | 104 | 3.6% |

Rhetorical questions are the rarest by an order of magnitude, which
matters when interpreting per-subtype performance: small-N estimates
on rhetorical questions are noisy.

### 2.3 Success criteria

Our primary success criterion is **strict success**, defined on the
hand-labeled human evaluation set:

```
strict_success = sarcasm_removed ∧ ¬meaning_change
```

Both conditions are binary labels assigned by annotators (and we have
two independent annotators per sample). A model achieves strict
success on a sample only if both annotators agree the sarcasm was
removed AND both agree the meaning was preserved.

This is a deliberately conservative criterion. A model can have a high
flip rate (sarcasm removed) but a high meaning-change rate (because
removal was achieved by deletion or paraphrasing into nonsense) and
still score poorly. T5-Joint (43.6%) wins because it achieves both
conditions simultaneously.

---

## 3. Dataset Construction

### 3.1 Source: News Headlines Dataset for Sarcasm Detection (NHDSD)

**[Misra (2019)](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)** —
28,619 headlines split between two professional newsroom-style
sources:

| Source | Type | Count |
|---|---|---|
| TheOnion | Sarcastic (intentionally) | 13,634 |
| HuffPost | Non-sarcastic (newsroom) | 14,985 |

We chose NHDSD over the more common Twitter-derived sarcasm corpora
for three reasons:

- **Professional writing** — no spelling noise, grammatical mistakes, or
  abbreviations to normalize away
- **Self-contained** — headlines stand alone; no reply-thread or quoted
  context required
- **Reliable labels at the source level** — TheOnion's sole purpose is
  sarcastic news; HuffPost is straightforward reporting. The labels are
  *purpose-driven*, not crowdsourced post-hoc

After dedup and whitespace normalisation we end up with 28,497 unique
headlines.

### 3.2 Data Quality Audit

We were uncertain how reliable NHDSD's labels really are at the row
level — TheOnion publishes some non-sarcastic operations posts, and
HuffPost occasionally posts opinion pieces with sarcastic framing. So
we ran an audit:

1. **Relabel with StepFun 3.5 Flash**: every headline gets a fresh
   binary classification from a strong LLM. Agreement with NHDSD's
   labels is **80.19%** — substantially below what you'd expect from a
   "high-quality, purpose-built" dataset.
2. **Cross-validate disagreements with Nemotron 3 Nano 30B**: of the
   5,644 disagreements, **4,076 are confirmed as suspected NHDSD
   mislabels** by a second independent LLM (72.2% confirmation rate).

These 4,076 are browsable in the webapp's `/mislabels` page with
direct links to the original articles so a human can judge for
themselves.

**Important caveat**: we did **not** silently overwrite NHDSD's labels.
The main training pipeline still uses the raw NHDSD labels because:

- The cross-validation is itself an audit signal that uses two LLMs
  whose own reliability we question (and end up disproving — see
  Finding 7.1)
- "Two LLMs disagreed with the original label" is a flag for
  inspection, not a substitute for ground truth
- Quietly rewriting the dataset would make our results
  non-reproducible relative to other NHDSD-based work

Treating LLM cross-validation as ground truth would be the same
methodological mistake we diagnose later in the classifier audit.

### 3.3 Synthetic Parallel Corpus

No large-scale sarcastic↔non-sarcastic paired dataset exists at the
scale we need. We construct one with StepFun 3.5 Flash via OpenRouter:

| Direction | Pairs Generated |
|---|---|
| Sarcastic → Non-sarcastic (from TheOnion sources) | 13,588 |
| Non-sarcastic → Sarcastic (from HuffPost sources) | 14,948 |
| **Total raw pairs** | **28,536** |

Generation is run at temperature 0.8 with a few-shot prompt that
shows three sarcastic↔non-sarcastic examples. The full prompts are in
[Appendix B](#appendix-b--llm-annotation-prompts). We discard
9 headlines blocked by content filters.

The non→sarcastic pairs additionally include a strategy label
(`<sarcasm>`, `<irony>`, `<satire>`, etc.) selected by the LLM during
generation.

### 3.4 Strategy Augmentation

For every non→sarcastic pair we additionally generate **5 strategy
variants** (one per missing subtype from the 6 iSarcasm categories),
producing **89,688 strategy-annotated records** with perfect class
balance:

| Subtype | Records |
|---|---|
| sarcasm | 14,948 |
| irony | 14,948 |
| satire | 14,948 |
| overstatement | 14,948 |
| understatement | 14,948 |
| rhetorical_question | 14,948 |

**Why augment?** Two reasons. First, the subtypes are the basis of the
6-way ablation study — we need balanced subtype coverage so that the
ablation is fair. Second, the joint-task model (T5-Joint) needs
explicit strategy supervision for every training example, and the
original LLM-selected strategies are skewed toward `sarcasm` and
`irony` (because StepFun defaults to those). Augmentation produces a
clean per-subtype training signal.

### 3.5 Train/Val/Test Splits

Two independent splits power the downstream experiments:

| Split | Used by | Train / Val / Test |
|---|---|---|
| `data/splits/sar_to_non/` | BART-Base, BART-RL | 10,868 / 1,356 / 1,364 |
| `data/splits/sar_to_non_context_enhanced/` | BART-CE, BART-CE+RL, both LLaMA variants | 8,258 / 1,029 / — |
| `data/joint_and_ablate_prepared/` | T5-Joint, T5-Control, 6 ablations | 80/10/10 stratified by subtype |

Two important properties:

1. **Source-level splits prevent leakage** — all six strategy variants
   of a given source headline land in the same split. Without this,
   the model can memorise a source headline from one variant and "do
   well" on its other variants in the test set.
2. **Stratified by subtype** — train/val/test each contain proportional
   representation of all six subtypes, so per-subtype eval is fair.

### 3.6 Golden Human-Eval Set

A separate **140-sample golden set** is hand-labeled by **2 independent
annotators** for the human evaluation:

- Stratified across the six subtypes (proportional to the test set,
  so rhetorical questions are still under-represented at N=14)
- Three models are included: **T5-Joint**, **T5-Control**, **BART-RL**
  — chosen as the most informative comparison set (joint vs control
  isolates the joint objective; BART-RL is the alternate winner
  candidate by automated metrics)
- Two binary labels per sample: `sarcasm_removed` and `meaning_change`
- Inter-annotator κ across the three models: **0.839, 0.883, 0.884** —
  excellent agreement (Landis & Koch interpretation)

The golden data lives in `data/golden/cleaned/` and the analysis
script that produces the merged + per-subtype + classifier-comparison
breakdowns is `scripts/analyze_golden_results.py`.

### 3.7 Limitations of the synthetic data

Three honest acknowledgements:

1. **The corpus is shaped by StepFun's biases.** Whatever stylistic
   tics, vocabulary preferences, or topic blind spots StepFun has
   propagate into our training data. A model trained on this corpus
   will reproduce those biases.
2. **The "non-sarcastic" targets are also synthetic.** The neutral
   rewrites are produced by the same LLM that generated the sarcastic
   versions; they're not actual newsroom-quality writing. Models that
   match these targets too closely (high BLEU vs reference) may not
   generalise to real-world neutral text.
3. **Strategy labels are LLM-assigned.** We don't have human-validated
   strategy labels for every training example — only the cross-
   validation between StepFun and Nemotron, and the 140-sample golden
   set. A finer-grained subtype taxonomy would require its own
   annotation effort.

These limitations bound the conclusions we can draw and motivate
several items in the [Future Work](#9-future-work) section.

---

## 4. Model Recipes

### 4.1 Recipe 1 — BART supervised fine-tuning (Yang Zhi)

**Models**: BART-Base, BART-CE
**Script**: [`scripts/train.py`](../scripts/train.py)
**Backbone**: `facebook/bart-base` (140M parameters)

BART is an encoder-decoder pretrained with a denoising objective —
spans of the input are masked and the decoder reconstructs them. This
makes it natural for style transfer because the task structurally
resembles denoising: corrupt the input (with sarcasm) and reconstruct
it (without).

The two BART variants share recipe but differ in training data:

- **BART-Base** trains on `data/splits/sar_to_non/` (10,868 pairs) —
  the main supervised split with no extra context
- **BART-CE** trains on `data/splits/sar_to_non_context_enhanced/`
  (8,258 pairs) — a smaller subset where each pair is augmented with
  the scraped article body, so the model can ground the rewrite in the
  underlying news event

CE = "Context-Enhanced", not "Cross-Entropy". (We confused this on
ourselves more than once during the project.)

```bash
python scripts/train.py \
    --model facebook/bart-base \
    --direction sar-to-non \
    --epochs 5 \
    --batch_size 16 \
    --lr 3e-4
```

| Hyperparameter | Value |
|---|---|
| Trainer | HuggingFace `Seq2SeqTrainer` |
| Epochs | 5 (early stop, patience 2) |
| Batch size | 16 |
| Learning rate | 3e-4 |
| Max sequence length | 128 tokens |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Best metric | BLEU on validation |
| Precision | bf16 (CUDA) |
| Input format | Raw sarcastic headline (no prefix) |
| Seed | 42 |

BART is pretrained with its own denoising objective and **does not**
expect a task prefix, unlike T5. Adding one degrades performance
slightly because the model has never seen prefixes during pretraining.

### 4.2 Recipe 2 — T5 supervised fine-tuning (Camille's separate repo)

**Models**: T5-Joint, T5-Control, T5-Joint (small), six ablations
**Script**: [`finetune_T5.py`](https://github.com/camille-readbean/CS4248-project-AY2526S2/blob/main/scripts/finetune_T5.py)
**Orchestration**: SLURM via [`slurm_finetune_t5.sh`](https://github.com/camille-readbean/CS4248-project-AY2526S2/blob/main/scripts/slurm_finetune_t5.sh)
**Backbone**: `google-t5/t5-base` (220M parameters); the older `joint`
variant uses `t5-small` (60M)

The T5 family was trained in **a separate repo** with its own pipeline
and a different recipe than the BART side. Camille
([`camille-readbean/CS4248-project-AY2526S2`](https://github.com/camille-readbean/CS4248-project-AY2526S2))
implemented the joint task and ran the experiments on the NUS SLURM
cluster; the resulting checkpoints feed back into our shared eval
pipeline alongside the BART models.

| Hyperparameter | Value |
|---|---|
| Trainer | `Seq2SeqTrainer`, `predict_with_generate=True` |
| Epochs | 4 (no early stopping) |
| Per-device batch | 8 |
| Grad accumulation | 2 (effective batch 16) |
| Learning rate | 3e-4 |
| Scheduler | Cosine, `warmup_ratio=0.06` |
| Weight decay | 0.01 |
| Max source/target length | 1248 tokens |
| Best metric | `eval_loss` |
| Precision | fp16 |
| Compute | 1× NV GPU, 32G mem, SLURM `gpu-long`, 5h limit |
| Seed | 42 |

The three task variants differ in input/target format only:

| Variant | Input prefix | Target |
|---|---|---|
| **T5-Joint** | `"rewrite to non-sarcastic and predict strategy: "` | `"strategy: {strategy} rewrite: {non_sarcastic}"` |
| **T5-Control** | `"rewrite to non-sarcastic: "` | `{non_sarcastic}` (plain) |
| **6 Ablations** | `"rewrite to non-sarcastic: "` | `{non_sarcastic}` (plain) |

**T5-Joint is the best model overall on human evaluation** (43.6%
strict success vs T5-Control's 39.3%). Same data, same recipe, same
backbone — the only difference is whether the target string includes
the strategy token.

#### Why does T5-Joint win?

The strategy-prediction prefix forces *task decomposition* before
generation. Instead of learning a one-shot mapping from sarcastic to
non-sarcastic, the model learns:

1. Identify the sarcastic mechanism (which subtype is this?)
2. Conditional on the mechanism, plan how to remove it
3. Generate the rewrite

This decomposition is implicit — there's no architectural change, just
a longer target string — but it improves meaning preservation
substantially (16.4% meaning change vs T5-Control's 25%, on the same
hand-labeled set).

#### Ablation construction

`prepare_t5_datasets.py` builds the six ablation splits as follows:

1. Take the full strategy-augmented training set
2. For each of the six subtypes:
   - Drop all training and validation examples with that subtype
   - Stratified-downsample the remaining pool to the *minimum across
     all six drops* (so every ablation trains on the same number of
     examples — making the comparison fair)
3. The test set is the **full held-out split**, shared across all six
   ablations. Test-time the model still sees all six subtypes.

This design lets us isolate the contribution of each subtype: if
dropping `irony` from training made the model significantly worse on
`irony` at test time, that would tell us irony is unique. Spoiler
(Finding 7.5): it doesn't.

### 4.3 Recipe 3 — Reinforcement learning (BART variants)

**Models**: BART-RL, BART-CE+RL
**Script**: [`scripts/train_rl.py`](../scripts/train_rl.py)
**Initialisation**: SFT BART checkpoint as both policy and frozen
reference

The RL recipe takes a BART SFT checkpoint and refines it using
**REINFORCE with a KL penalty against the frozen SFT reference**. This
is the recipe ViSP (Zhu et al. 2025) used for sarcasm *generation*,
inverted here for sarcasm *removal*.

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────┐
│ SFT BART    │ ──► │ Sample with  │ ──► │ Sarcasm        │ ──► │ REINFORCE│
│ (policy π)  │     │ top-k/top-p  │     │ classifier     │     │ + KL     │
└─────────────┘     └──────────────┘     │ + ROUGE-L      │     └──────────┘
       ▲                                  └────────────────┘          │
       │            KL(π || π_ref)                                    │
       └──────────────────────────────────────────────────────────────┘
                          frozen reference
```

#### Loss formulation

```
r       = α · (1 − P_sarcastic(output)) + (1 − α) · ROUGE-L(output, ref)
L_rl    = −(r − baseline) · Σ log π_θ(y | x)
L_total = L_rl + β · KL(π_θ || π_ref)
```

| Symbol | Meaning | Value |
|---|---|---|
| `α` | Style reward weight | 0.5 |
| `β` | KL penalty coefficient | 0.2 |
| `P_sarcastic` | Sarcasm classifier probability | from a held-out classifier |
| `ROUGE-L` | Longest common subsequence overlap | output vs reference target |
| `baseline` | Reward baseline | EMA of batch reward, decay 0.9 |

**Why a composite reward?** Pure style reward
(`r = 1 − P_sarcastic`) saturates immediately because the SFT outputs
already classify as ~1.0 non-sarcastic on the reward model. The
ROUGE-L term provides the content signal — without it, the policy
collapses to "delete everything" because a blank output still scores
high on the classifier.

| Hyperparameter | Value |
|---|---|
| Learning rate | 1e-5 (10× lower than SFT — drift risk) |
| Epochs | 3 |
| Per-device batch | 8 |
| Sampling | top-k=50, top-p=0.95, temp=0.8 |
| Gradient clipping | max_norm=1.0 |
| Reward baseline | EMA, decay 0.9 |
| Seed | 42 |

**Known failure mode — reward hacking**: even with the ROUGE-L term
and the KL penalty, BART-RL learns a different shortcut. Rather than
deleting tokens entirely, it deletes *just the sarcastic words* while
keeping enough overlap with the reference to satisfy ROUGE-L. The
classifier sees a non-sarcastic output, ROUGE-L sees enough overlap,
the KL penalty can't tell anything's wrong because the policy is still
producing fluent text — but a human reading the output sees the
meaning has been gutted. This is the central cautionary finding of the
project (Finding 7.3).

### 4.4 Recipe 4 — LoRA instruction tuning (LLaMA)

**Models**: LLaMA 3.2 1B, LLaMA 3.2 1B (context)
**Scripts**:
[`train_llama.py`](../scripts/train_llama.py),
[`train_llama_context.py`](../scripts/train_llama_context.py)
**Backbone**: `meta-llama/Llama-3.2-1B-Instruct` (1.24B parameters)

LLaMA is decoder-only and instruction-tuned, so the recipe diverges
from the seq2seq pipeline. We use **PEFT LoRA adapters** on every
attention + MLP projection layer, keeping the base weights frozen.
~6M of 1.24B parameters are trainable (0.5%). The chat template wraps
a fixed system prompt and the user message; loss is masked to the
assistant response only (prompt tokens set to −100 in labels).

**System prompt** (verbatim from the training script):

> You are a writing assistant. Rewrite sarcastic news headlines as
> neutral, factual equivalents that preserve the core meaning without
> irony or mockery. Respond with only the rewritten headline, no
> explanation.

| Hyperparameter | Value |
|---|---|
| LoRA rank `r` | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable params | ~6M of 1.24B (0.5%) |
| Learning rate | 2e-4 |
| Batch × grad accum (base) | 8 × 2 = 16 effective |
| Batch × grad accum (context) | 4 × 4 = 16 effective |
| Epochs | 3 |
| Max length (base) | 256 tokens |
| Max length (context) | 1024 tokens |
| Scheduler | Cosine, 5% warmup |
| Precision | bf16 + gradient checkpointing |

The two variants:

- **LLaMA 3.2 1B**: headline-only prompt
- **LLaMA 3.2 1B (context)**: prompt includes the scraped article body
  (max length is bumped to 1024 to fit, batch drops accordingly)

After training, the LoRA adapter is merged back into the base model
and exported to **GGUF format** so [LMStudio](https://lmstudio.ai/)
can serve it on consumer hardware. The merged model is also uploaded
to HuggingFace under
[`SeeYangZhi/llama-3.2-1b-sarcasm-rewriter`](https://huggingface.co/SeeYangZhi/llama-3.2-1b-sarcasm-rewriter).

### 4.5 Why these architectural choices?

| Choice | Justification |
|---|---|
| BART-base (140M) | Strong denoising prior makes it a natural seq2seq baseline; tiny enough to train on a single consumer GPU |
| T5-base (220M) | Designed for prefix-conditioned tasks; the "task prefix" pretraining objective is exactly what the joint task exploits |
| LLaMA 3.2 1B | Largest model we can serve in LMStudio on consumer hardware; lets us test whether the capacity gap matters |
| LoRA over full fine-tuning | 0.5% trainable params, much lower memory, identical recipe across variants |
| REINFORCE not PPO | Simpler implementation, comparable results for this task scale, doesn't require a value head |
| KL against frozen reference | Standard regulariser to prevent policy drift — though Finding 7.3 shows it's not sufficient on its own |

We deliberately did **not** try GPT-2. Earlier project iterations
considered it; we abandoned it because (a) it lacks an instruction-
tuned chat template, (b) its perplexity on news-style text is
substantially worse than BART/T5, and (c) running four families on
fourteen models was already a stretch on the team's compute budget.

---

## 5. Evaluation Methodology

### 5.1 7-Metric Automated Pipeline

`scripts/eval_pipeline.py --multi_classifier` produces seven
complementary metrics for every model output. The webapp
[`/eval`](../webapp/frontend/src/app/eval/page.tsx) page documents
each metric in the same depth, with score interpretation tables and
the limitations.

#### Metric 1: Sarcasm Flip Rate × 3 classifiers

**Question**: does the classifier think sarcasm was removed?

We run **three** off-the-shelf sarcasm classifiers from different
training domains:

| Classifier | Training data | HuggingFace ID |
|---|---|---|
| RoBERTa-Twitter | Twitter irony | [`cardiffnlp/twitter-roberta-base-irony`](https://huggingface.co/cardiffnlp/twitter-roberta-base-irony) |
| Bert-Kaggle | Kaggle headlines | [`helinivan/english-sarcasm-detector`](https://huggingface.co/helinivan/english-sarcasm-detector) |
| RoBERTa-News | News headlines | [`jkhan447/sarcasm-detection-RoBerta-base-POS`](https://huggingface.co/jkhan447/sarcasm-detection-RoBerta-base-POS) |

For each classifier and each output:
- `hard_flipped = 1` if the input is classified as sarcastic AND the
  output is classified as non-sarcastic (the classifier "flipped" its
  judgment)
- `flip_rate = mean(hard_flipped)` across the test set

**Why three?** Single-classifier flip rate is a moving target. Same
output, different classifier, different answer — see Finding 7.1 for
the data showing 33pp spread on the same outputs. Using all three
exposes the disagreement instead of hiding it.

#### Metric 2: Semantic Similarity

**Tool**: `sentence-transformers/all-MiniLM-L6-v2`
**Question**: is the core meaning preserved between input and output?

Cosine similarity between sentence embeddings of the input and output.
Score interpretation:

| Score | Interpretation |
|---|---|
| 0.95+ | Nearly identical (possibly just paraphrased) |
| 0.85–0.95 | Good meaning preservation ✓ |
| 0.70–0.85 | Moderate drift |
| < 0.70 | Significant meaning loss ✗ |

**Limitation**: doesn't detect copying. A model that just lowercases
the input scores 0.99. Always pair with BLEU vs input.

#### Metric 3: Perplexity

**Tool**: GPT-2 reference language model
**Question**: is the output fluent, natural English?

Token-level perplexity of the output under GPT-2. Catches degenerate
outputs (truncations, gibberish, broken syntax) that other metrics
might miss.

| Score | Interpretation |
|---|---|
| < 300 | Very fluent |
| 300–600 | Normal |
| 600–1000 | Somewhat disfluent |
| > 1000 | Problematic |

**Limitation**: mean perplexity is dominated by long-tail outliers — a
single broken sample drags the average up substantially. The dashboard
reports the mean (consistent with `eval_pipeline.py`) but flag outliers
rather than treating the absolute value as authoritative.

#### Metric 4: BLEU vs Input

**Tool**: `sacrebleu`
**Question**: how much n-gram overlap is there between the output and
the input?

Note this is BLEU vs **input**, not BLEU vs reference. The point is to
detect **copying** vs **genuine rewriting**.

| Combination | Interpretation |
|---|---|
| High BLEU + High similarity | Paraphrasing — minimal real change |
| Low BLEU + High similarity | Genuine rewriting ✓ |
| Low BLEU + Low similarity | Meaning lost ✗ |

**Limitation**: only meaningful in combination with similarity. Low
BLEU alone could mean either successful rewriting or content
destruction.

#### Metric 5: Edit Distance

**Tool**: word-level Levenshtein, normalised to [0, 1]
**Question**: how much was the text modified?

| Score | Interpretation |
|---|---|
| 0.0–0.3 | Minor edits (punctuation, casing) |
| 0.4–0.6 | Moderate rewriting |
| 0.7–0.9 | Significant rewriting |
| 0.9+ | Complete rewrite |

Complementary to BLEU — measures structural change rather than n-gram
overlap. Useful for separating models that delete tokens (high edit
distance, low BLEU) from models that paraphrase (moderate edit
distance, moderate BLEU).

#### Metric 6: LLM-as-Judge

**Tool**: Gemini 2.5 Flash, sampled subset (50 samples per model)
**Question**: what does a strong LLM think across three dimensions?

Each output is scored 1–5 on:

- `sarcasm_removed` — is the output non-sarcastic?
- `meaning_preserved` — is the core claim intact?
- `fluency` — is it natural English?

**Limitation**: LLM judges are known to be biased toward LLM-style
outputs (Zheng et al. 2023). We use this as a sample evaluation, not
a primary metric.

#### Metric 7: Paraphrase Score

**Formula**: `paraphrase_score = similarity × (1 − BLEU_vs_input)`

Existing metrics fail individually:

| Scenario | Similarity | BLEU vs input | Problem |
|---|---|---|---|
| Just lowercase input | 0.99 | 0.95 | Looks good but no real change |
| Complete rewrite | 0.65 | 0.02 | Looks bad but might be necessary |

The paraphrase score multiplies the two so a model has to score well
on **both** to win.

**Concrete example**:
- Input: *"Man Shocked By Obvious Fact"*
- Output A: *"man shocked by obvious fact"* → similarity 0.99,
  BLEU 0.95 → paraphrase **0.05** (just copied)
- Output B: *"A person was surprised to learn something widely known"*
  → similarity 0.85, BLEU 0.08 → paraphrase **0.78** (genuine rewrite)

Score interpretation:

| Score | Interpretation |
|---|---|
| > 0.20 | Good — high similarity + low copying |
| 0.10–0.20 | Moderate |
| < 0.05 | Poor — either copying or meaning lost |

This is the metric we lead with on the dashboard for "best model by
genuine rewriting".

### 5.2 Multi-Classifier Audit

Beyond reporting all three classifier flip rates, we compute the
**spread** (max − min across classifiers) for each model. If the spread
is high (>20pp), no single classifier number can be trusted. If it's
low, there's at least convergence.

**Result preview** (full table in §6.3): 12 of 14 models show >30pp
spread. Only LLaMA 3.2 1B and BART-CE show low spread, and that's
because both models are aggressive rewriters that destroy enough of
the input to confuse all three classifiers symmetrically.

We also compute per-model κ scores against human ground truth on the
golden set, producing a 9-cell breakdown (3 models × 3 classifiers)
that's the centrepiece of Finding 7.1.

### 5.3 Human Evaluation Methodology

For three models (T5-Joint, T5-Control, BART-RL) we hand-label **140
stratified samples × 2 independent annotators**.

#### Annotation procedure

Each sample is presented to the annotator as:

```
INPUT:  <sarcastic headline>
OUTPUT: <model rewrite>

Q1: Did the model remove the sarcasm? [Y / N]
Q2: Did the meaning change? [Y / N]
```

Annotators do not see which model produced which output. They see all
three model outputs for each input but in randomised order across
annotators.

#### Why two annotators?

Single-annotator labels are noisy. Two annotators let us compute
**inter-annotator κ**, which is the standard reliability check for
binary annotation:

| Model | Inter-annotator κ |
|---|---|
| T5-Joint | 0.839 |
| T5-Control | 0.883 |
| BART-RL | 0.884 |

Landis & Koch (1977) interpretation: κ > 0.80 is "almost perfect
agreement". Our annotators agree at this level on all three models,
which means the labels are **reliable ground truth** — and dramatically
more so than any of the three classifiers.

#### Strict success definition

```
strict_success = sarcasm_removed_by_both_annotators
                  AND
                  meaning_unchanged_by_both_annotators
```

A sample only counts as a success if both annotators agreed sarcasm
was removed AND both agreed meaning was preserved. This is the
primary metric we use to rank models in human evaluation.

### 5.4 What we evaluate where

| Eval | Coverage | Purpose |
|---|---|---|
| 7-metric pipeline | All 14 models × 2,857 test samples | Primary automated comparison |
| 3-classifier audit | All 14 models | Show classifier disagreement |
| Human evaluation | 3 models × 140 samples × 2 annotators | Ground truth for "which model is actually best" |
| LLM-as-judge | 14 models × 50-sample subset | Sample sanity check |
| Per-subtype breakdown | All 14 models × 6 subtypes | Show where each model fails |

The combination — multi-classifier + human + per-subtype — is what
makes the comparison robust. Any single signal could be gamed; the
combination cannot.

---

## 6. Results

### 6.1 Headline numbers

**Best model overall (by human evaluation)**: T5-Joint

| Metric | T5-Joint | T5-Control | BART-RL |
|---|---|---|---|
| **Strict success** | **43.6%** | 39.3% | 34.3% |
| Human flip rate | 54.3% | 54.3% | 52.9% |
| Meaning change rate | **16.4%** | 25.0% | 40.7% |
| Inter-annotator κ | 0.839 | 0.883 | 0.884 |

T5-Joint and T5-Control remove sarcasm at identical rates (54.3%); the
difference is meaning preservation. T5-Joint changes the meaning of
16.4% of its outputs, T5-Control 25%, BART-RL 40.7%.

### 6.2 Full Automated 7-Metric Summary (All 14 Models)

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

†Flip rate shown for RoBERTa-Twitter only — see §6.3 for the spread
across all three classifiers.

### 6.3 Multi-Classifier Audit

Same model, three classifiers, three different stories:

| Model | RoBERTa-Twitter | Bert-Kaggle | RoBERTa-News | Spread |
|---|---|---|---|---|
| t5_base_joint | 8.4% | 41.6% | 21.2% | **33.2 pp** |
| t5_control | 8.1% | 40.9% | 20.3% | 32.9 pp |
| joint | 8.4% | 41.1% | 22.4% | 32.7 pp |
| bart_base_rl | 8.8% | 39.9% | 26.4% | 31.1 pp |
| bart_base | 8.4% | 41.8% | 21.7% | **33.4 pp** |
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

**12 of 14 models show >30pp spread.** Pick whichever classifier
supports your hypothesis — that's why we cross-check with human eval.

### 6.4 Classifier vs Human (Cohen's κ)

| Classifier | Avg κ vs human (across 3 models) | Best κ in matrix | Worst κ in matrix |
|---|---|---|---|
| RoBERTa-Twitter | +0.019 | +0.044 | −0.005 |
| Bert-Kaggle | **−0.075** | −0.055 | **−0.113** |
| RoBERTa-News | +0.104 | **+0.179** | +0.055 |
| **Human inter-annotator** | **0.836–0.884** | — | — |

**4 of 9 model×classifier cells show negative κ** — the classifier
*anti*-correlates with humans. Best κ across the entire matrix is +0.18
(RoBERTa-News on T5-Joint). Compare to the 0.84+ that humans achieve
agreeing with each other.

### 6.5 The 9-Cell Breakdown

| Model | Classifier | Clf Flip | Human Flip | Accuracy | κ |
|---|---|---|---|---|---|
| T5-Joint | RoBERTa-Twitter | 5.7% | 54.3% | 48.6% | +0.044 |
| T5-Joint | Bert-Kaggle | 9.3% | 54.3% | 43.6% | −0.055 |
| T5-Joint | RoBERTa-News | 24.3% | 54.3% | 57.1% | **+0.179** |
| T5-Control | RoBERTa-Twitter | 5.7% | 54.3% | 47.1% | +0.017 |
| T5-Control | Bert-Kaggle | 12.1% | 54.3% | 40.7% | **−0.113** |
| T5-Control | RoBERTa-News | 25.0% | 54.3% | 50.7% | +0.055 |
| BART-RL | RoBERTa-Twitter | 4.3% | 52.9% | 47.1% | −0.005 |
| BART-RL | Bert-Kaggle | 15.0% | 52.9% | 45.0% | −0.058 |
| BART-RL | RoBERTa-News | 31.4% | 52.9% | 52.9% | +0.077 |

The classifiers consistently *under*-detect flips relative to humans:
they say sarcasm was removed in 5–31% of cases, while humans say it
was removed in 53–54%. The classifiers are biased toward "still
sarcastic" because they were trained on classification, not removal
verification — they look at each output in isolation and ask "does
this look sarcastic?", which is the wrong question for our task.

### 6.6 Per-Subtype Performance (T5-Joint, full test set)

| Subtype | N | Flip Rate | Similarity | BLEU | Para Score |
|---|---|---|---|---|---|
| sarcasm | 883 | 7.5% | 0.867 | 0.213 | 0.193 |
| irony | 614 | 9.6% | 0.874 | 0.212 | 0.193 |
| satire | 525 | 6.1% | 0.888 | 0.222 | 0.204 |
| overstatement | 401 | 8.7% | 0.873 | 0.224 | 0.204 |
| understatement | 330 | 9.7% | 0.862 | 0.183 | 0.164 |
| rhetorical_question | 104 | 13.5% | 0.840 | 0.169 | 0.148 |

### 6.7 Per-Subtype Human Evaluation (T5-Joint, golden set)

| Subtype | N | Human Flip | Meaning Δ | Strict Success |
|---|---|---|---|---|
| sarcasm | 69 | 59.4% | 23.2% | 44.9% |
| irony | 30 | 50.0% | 6.7% | 50.0% |
| rhetorical_question | 14 | 50.0% | 0.0% | **50.0%** |
| understatement | 11 | 54.5% | 18.2% | 36.4% |
| satire | 9 | 55.6% | 33.3% | 22.2% |
| overstatement | 7 | 28.6% | 0.0% | 28.6% |

Sample sizes for satire and overstatement are small; treat those
percentages with appropriate caution.

### 6.8 Ablation Results

Six retrains of the T5 control recipe with one subtype dropped each:

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
No single subtype is load-bearing.

---

## 7. Findings & Analysis

### 7.1 Automated flip rate is not a valid primary metric

The 9-cell breakdown above is the centrepiece. To restate the headline:

- **Average κ vs human across all classifiers**: −0.075 to +0.104
- **Best κ across the entire matrix**: +0.18
- **Number of negative-κ cells**: 4 out of 9
- **Inter-annotator κ on the same data**: 0.84+

Two random untrained humans would average around κ = 0. Our best
classifier scores +0.18 — closer to "noise" than to "agreement". And
the worst (−0.11) is *anti-correlated* with humans, meaning you'd be
better off flipping its predictions.

#### Why do classifiers fail?

The surface-level answer is **domain mismatch**: a Twitter classifier
on news headlines, a Reddit-tone classifier on sarcastic news, etc.
The deeper answer is task mismatch:

> The classifiers were trained for **sarcasm detection** ("is this
> single text sarcastic?"). We are using them for **removal
> verification** ("did the rewrite become non-sarcastic *relative to*
> the input?"). These are fundamentally different tasks.

The classifier never sees the input/output pair together. It judges
each output in isolation. It can't detect *transformation*, only static
classification. Worse, sarcasm is often signalled by **cultural
context** ("Google Unveils AI That Can Finally Make Eye Contact" is
sarcastic because we know Google's actual capabilities), and the
classifier doesn't have that context — it sees a headline that *looks*
like a normal tech-news headline and judges it accordingly.

The implication for the field is uncomfortable: **any sarcasm style
transfer paper that reports a single classifier flip rate is reporting
a number that could be off by 30 percentage points depending on which
classifier was used.** The correct protocol is multi-classifier +
human evaluation. We adopt it; we recommend others do too.

### 7.2 T5-Joint wins via task decomposition

| Metric | T5-Joint | T5-Control | Δ |
|---|---|---|---|
| Human meaning change | **16.4%** | 25.0% | −8.6 pp |
| Human strict success | **43.6%** | 39.3% | +4.3 pp |
| LLM meaning preserved | **4.82** | 4.64 | +0.18 |

The two models share **everything**: same backbone (`t5-base`), same
hyperparameters, same training data, same recipe, same compute. The
only difference is the target string format:

- T5-Joint: `"strategy: irony rewrite: Local man strongly defends his interpretation of the Constitution"`
- T5-Control: `"Local man strongly defends his interpretation of the Constitution"`

Adding the strategy token to the target improves meaning preservation
by ~9 percentage points on the same hand-labeled set. Why?

#### Hypothesis: forced decomposition

The strategy token forces the model to identify the sarcastic
mechanism *before* generating the rewrite. Decoder timestep 1 has to
emit `strategy: <subtype>`, which means by the time the model gets to
generating the rewrite it has already implicitly answered "what's
sarcastic about this input?". Different subtypes require different
rewriting strategies:

- **Exaggeration / overstatement** → tone down the claim
- **Rhetorical question** → convert to declarative statement
- **Irony** → state the actual intended meaning
- **Satire** → strip the mock-news framing

Without the prefix, the model has to figure out *what's sarcastic* and
*how to fix it* and *what to keep* simultaneously. More cognitive load
→ more mistakes → more meaning drift.

**Analogy**: it's like asking someone to "edit this document" vs "fix
the grammar errors in this document". The second is more constrained
and produces more targeted changes.

#### Why don't BART/LLaMA replicate this?

We didn't try. The T5-Joint architecture is implemented in Camille's
separate repo and the joint task happens to be a natural fit for T5's
"task prefix" pretraining objective. Replicating it on BART would
require non-trivial code changes (different tokeniser, different
output parsing); on LLaMA it would require restructuring the chat
template. **This is the obvious follow-up**: see §9.

### 7.3 BART-RL is reward hacking

| Metric | BART-RL | T5-Joint | Δ |
|---|---|---|---|
| RoBERTa-News flip rate | 26.4% | 21.2% | +5.2 pp |
| Bert-Kaggle flip rate | 39.9% | 41.6% | −1.7 pp |
| **Human meaning change** | **40.7%** | 16.4% | **+24.3 pp** |
| Human strict success | 34.3% | 43.6% | −9.3 pp |

BART-RL's classifier flip rates are competitive — slightly higher than
T5-Joint on RoBERTa-News, slightly lower on Bert-Kaggle. By the
classifier metrics it's a credible best-model candidate. Human eval
demolishes that story: it changes the meaning of **40.7% of its
outputs**, more than double T5-Joint.

#### What's actually happening

The composite reward is

```
r = 0.5 · (1 − P_sarcastic(output)) + 0.5 · ROUGE-L(output, ref)
```

Both terms are satisfied by deletion:
- Deleting sarcastic words **reduces** `P_sarcastic` ✓
- Keeping enough non-sarcastic words **preserves** ROUGE-L overlap ✓
- But deletion ≠ rewriting

#### Concrete example

```
Input:    "Area Man Proud Of Completely Average Achievement"
BART-RL:  "Man achieves something."
```

The classifier is happy. ROUGE-L sees "man" and "achieve" overlap.
The human annotator marks the meaning destroyed — the irony about
*pride in averageness* is gone, not rewritten. The KL penalty against
the SFT reference (β = 0.2) isn't enough to stop this because the
policy is still producing fluent text, just shorter.

This is **reward hacking** in the textbook sense: the model finds a
shortcut that optimises the reward function without achieving the
actual goal. It's the central cautionary finding of the project, and
it's structurally analogous to the failure modes of large-scale RLHF
with imperfect reward models.

#### Could we have prevented this?

In hindsight, three possible fixes:

1. **Bigger β**: a stronger KL penalty would constrain the policy
   closer to the SFT reference. Trade-off: less learning from the
   reward signal at all.
2. **Length penalty**: penalise outputs shorter than some fraction of
   the input. Would block the deletion shortcut but feels ad-hoc.
3. **Better reward model**: use a classifier that *can* detect meaning
   change, not just sarcasm presence. The fundamental problem is that
   our reward model is the same kind of classifier whose unreliability
   we document in Finding 7.1.

The third option is the principled fix. We don't have a meaning-aware
sarcasm classifier; building one is a significant project in itself.
**See §9 future work.**

### 7.4 LLaMA and BART-CE rewrite too aggressively

| Model | Similarity | Edit Distance | LLM Meaning |
|---|---|---|---|
| LLaMA 3.2 1B | 0.66 | 0.95 | 3.34 / 5 |
| BART-CE | 0.64 | 0.92 | 3.52 / 5 |
| BART-CE+RL | 0.61 | 0.93 | 2.90 / 5 |

These models edit ~95% of the input and retain ~1% n-gram overlap.
That's *generation* with the input as a prompt, not *style transfer*.

#### Concrete examples

```
Input:  "Inconsiderate Wife Leaves Bathroom A Total Mess After Home Birth"
LLaMA:  "Mother of Two Gives Birth at Home"
```

Loses the absurdity entirely. The original is darkly comic about the
mismatch between "inconsiderate mess" and "home birth"; the rewrite
is a generic hospital headline.

```
Input:    "Fucker Has Nerve To Be 22 Years Old"
LLaMA:    "Local Man Arrested for Sexual Assault at 22"
```

**Hallucination**: the model invents a crime that doesn't exist in
the input. This is the worst possible failure mode for a content
moderation pipeline — it generates fabricated allegations.

#### Why does this happen?

LLaMA's larger backbone (1.24B params vs BART's 140M) gives it more
generative latitude. It "knows" how news headlines are structured, so
when asked to rewrite a sarcastic one it produces a plausible
alternative that fits the news genre — but the alternative isn't
constrained to preserve the original event. The instruction tuning
helps with format compliance ("respond with only the rewritten
headline") but not with faithfulness.

BART-CE has the same problem in a different flavour: it's trained on
the context-enhanced split where the LLM had article bodies to
produce *deeper rewrites* than the headline alone would justify. The
model learns to generate Onion-style news rewrites that are
*inspired by* the input rather than *transformations of* it.

The trade-off is real: aggressive rewriters comprehend better but
lose faithfulness. Conservative rewriters preserve faithfulness but
edit too lightly. T5-Joint sits in the middle — high similarity
(0.87), moderate edit distance (0.63) — and that's why it wins.

### 7.5 Subtype ablations are interchangeable

Six retrains of the T5 control recipe with one subtype dropped each.
All six cluster within **0.005 similarity** of the unablated control:

| Held out | Similarity Δ from control |
|---|---|
| sarcasm | +0.005 |
| irony | +0.004 |
| satire | +0.002 |
| overstatement | +0.007 |
| understatement | +0.003 |
| rhetorical_question | +0.003 |

(Slight *improvements* are noise — within the test-set variance
across runs.)

#### Interpretation

Sarcasm subtypes share underlying mechanisms: hyperbole appears in
both sarcasm and overstatement; contradiction underlies both irony
and satire; rhetorical questions often contain embedded irony. The
model learns *generic patterns* from the five remaining subtypes
that transfer to the sixth.

This is a **positive generalisation finding**. It means the joint
model isn't memorising subtype-specific tricks; it's learning
something that generalises. It also means we can't trivially improve
the model by oversampling one subtype — there's no missing knowledge
to backfill.

The corollary: if you wanted to genuinely improve sarcasm style
transfer, you would **not** bet on more strategy data. You would bet
on either (a) a more capable backbone, (b) a better task formulation
(like T5-Joint), or (c) a meaning-aware reward signal for RL. The
data is not the bottleneck.

### 7.6 Different subtypes fail for different reasons

| Subtype | Classifier Miss Rate† | Why |
|---|---|---|
| satire | 80% | Mimics legitimate news format |
| rhetorical_question | 71% | Sarcasm in pragmatics, not lexicon |
| irony | 67% | Contradiction is contextual, no surface markers |
| understatement | 50% | Requires world knowledge of "appropriate" response |
| sarcasm (generic) | 44% | Has detectable lexical patterns |
| overstatement | 100%* | **Model failure** — only 28.6% human flip rate |

†% of human-labeled flips that the classifier failed to detect.

*Overstatement is the anomaly. 100% miss rate is misleading: it's
not that the classifier is bad, it's that **the model fails to
remove sarcasm in the first place**. Human flip rate on overstatement
is only 28.6% — humans agree the model didn't do its job. Why?

> Overstatement is **hard to rewrite without losing the core claim**.
> The claim *is* the exaggeration. If you remove the exaggeration
> from "Area Man Has Most Important Day Of His Life", you get
> "Area man has a day", which has no remaining content. The model's
> dilemma: keep the exaggeration → still sarcastic; remove too much
> → no meaning left.

This is a fundamentally harder subtype than the others, and our
training data doesn't have a good answer for it because the human
"correct" rewrites in the synthetic corpus also struggle with the
same dilemma.

### 7.7 Cumulative narrative

If we had to summarise the whole project in one sentence:

> **Strategy-aware joint training (T5-Joint) beats reinforcement
> learning with a classifier reward (BART-RL) because the latter
> optimises a metric we cannot trust, while the former forces the
> model to do something we can verify.**

Every other finding is corollary: the classifiers are unreliable
(7.1), so the RL reward is unreliable (7.3); the joint task forces
decomposition (7.2) which is verifiable; bigger backbones don't fix
the underlying problem (7.4); subtype data isn't the bottleneck
(7.5); and the hardest subtypes are model failures, not data
failures (7.6).

---

## 8. Limitations

We surfaced several limitations inline; collected here for visibility.

### Domain specificity

Every model in this project is trained on news headlines from
TheOnion and HuffPost. Sarcasm is strongly domain-dependent. A model
trained on news headlines will not transfer well to:

- Twitter (different lexical conventions, reply context)
- Reddit (longer-form, community-specific tone)
- Customer support tickets (sarcasm-as-frustration, not sarcasm-as-comedy)
- Spoken transcripts (paralinguistic cues we have no access to)

Cross-domain evaluation would likely show large degradation. We did
not run it because the project's scope is news headlines; this is a
known limit of the conclusions.

### Sample size on rare subtypes

Per-subtype counts in the human eval are heavy-tailed:

| Subtype | Golden N |
|---|---|
| sarcasm | 69 |
| irony | 30 |
| rhetorical_question | 14 |
| understatement | 11 |
| satire | 9 |
| overstatement | 7 |

Statistics for satire (N=9) and overstatement (N=7) have wide
confidence intervals. We report them for completeness but treat them
as suggestive, not definitive. Future work should oversample the rare
subtypes during golden set construction.

### Reward model fragility

The RL reward depends on a single off-the-shelf sarcasm classifier
whose unreliability we document. Even if BART-RL "worked" in the
sense of not reward-hacking, its outputs would only be as good as
that classifier. Building a more reliable reward signal — possibly
via human preference data, possibly via a meaning-aware verifier — is
a significant open problem.

### Synthetic data biases

Both the inputs and the targets in our training corpus were generated
by StepFun 3.5 Flash. Whatever stylistic preferences and topic blind
spots StepFun has propagate into our models. A model trained on
StepFun-generated targets will reproduce StepFun's idea of "neutral
news writing", which may not match actual newsroom writing.

### Eval metric coverage

The 7-metric pipeline is broad but not exhaustive. Notable gaps:

- **Factual accuracy**: we don't verify that the rewritten headline is
  factually correct relative to the underlying news event
- **Toxicity**: we don't check whether rewriting might introduce or
  amplify offensive content
- **Style consistency**: we don't measure whether the rewrite matches
  the style of real neutral news headlines vs LLM-generated ones
- **Length**: we don't penalise outputs that are dramatically longer
  or shorter than the input

A production system would need all four. Our scope is research
methodology, not deployment.

### Compute constraints

We trained on consumer-grade GPUs and a single SLURM partition (NUS
gpu-long, 5h time limit). This bounds:

- Maximum model size (LLaMA 3.2 1B is the largest we could fit + run
  ablations on within budget)
- Number of seeds per model (we report single-seed numbers; multi-
  seed would tighten the confidence intervals)
- Hyperparameter sweep depth (we mostly used defaults from the
  reference implementations)

A larger compute budget would let us run e.g. 5 seeds × 14 models and
report mean ± stddev, which would be more rigorous. Future work.

---

## 9. Future Work

Six concrete follow-ups, ordered by what we'd do first:

### 9.1 Joint task on BART and LLaMA backbones

The T5-Joint result (Finding 7.2) is suggestive but only proven on
T5. Is the joint-task gain a property of the task formulation (which
would generalise) or of T5's prefix-conditioning pretraining (which
wouldn't)? Replicating T5-Joint on BART and LLaMA — same training
data, same target format — would isolate the answer. Highest-impact
single follow-up.

### 9.2 DPO instead of REINFORCE

Direct Preference Optimization (Rafailov et al. 2023) replaces the
explicit reward model with a binary preference signal, which sidesteps
the reward-hacking failure mode. Pairs of outputs (one preferred, one
not) can come from human annotators or a pseudo-oracle (e.g.,
T5-Joint's outputs as preferred over BART-RL's). DPO is also more
stable to train than REINFORCE.

### 9.3 Domain-general sarcasm classifier

The current classifiers fail because they were trained on classification,
not removal verification, and on different domains than ours. A
purpose-built classifier — trained on (input, output, removed?)
triples on news headlines — would give us a meaningful reward signal
for RL and a meaningful auxiliary metric for evaluation. Constructing
the training data is the hard part; the classifier itself would be
straightforward.

### 9.4 Hallucination mitigation for LLaMA

LLaMA produces genuine rewrites at the cost of factual reliability
(Finding 7.4). Constrained decoding, retrieval-augmented generation
(grounding the rewrite in the original article), or a verification
pass with a separate model could all reduce hallucination. The
context-enhanced LLaMA variant is a step in this direction but
doesn't fully solve the problem.

### 9.5 Richer human evaluation

The current golden set has two binary labels per sample. Three
extensions would be valuable:

- **Subtype-level rationales**: ask annotators to explain *what's*
  sarcastic about the input, so we can see whether the model agrees
- **Pairwise comparison**: present two model outputs side-by-side
  ("which is the better rewrite?"), which is more reliable than
  absolute Likert scoring
- **Larger N for rare subtypes**: oversample rhetorical question,
  satire, and overstatement so per-subtype statistics are reliable

### 9.6 Cross-domain evaluation

Apply our best models (T5-Joint, possibly LLaMA) to sarcasm in:

- iSarcasmEval Twitter data (the original taxonomy source)
- Sarcasm Corpus V2 (Reddit)
- A human-written holdout we hand-construct for cross-domain testing

Quantify the degradation, then iterate.

---

## 10. Webapp

A live demo and interactive dashboard:
<https://github.com/SeeYangZhi/Project-LLMao>

### Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   Next.js Frontend  │────▶│   FastAPI Backend    │────▶│    LMStudio     │
│   (localhost:3000)  │     │   (localhost:8000)   │     │  (localhost:1234)│
│                     │     │                      │     │  LLaMA 3.2 1B   │
│  - Dashboard        │     │  - /api/metrics/*    │     └─────────────────┘
│  - Explorer         │     │  - /api/samples/*    │
│  - Playground       │     │  - /api/generate     │──▶ BART-CE+RL (local HF)
│  - Human Eval       │     │  - /api/human-eval/* │
│  - Pipeline / etc   │     │  - /api/mislabels/*  │
└─────────────────────┘     └──────────────────────┘
```

Local mode runs the FastAPI backend with live inference. Static mode
exports every metric and sample to JSON and serves them from a
pre-built Next.js bundle on Vercel — same React code, no backend
required. Live inference is disabled in static mode and the page
shows a banner explaining how to run it locally.

### Pages

| Route | What's there |
|---|---|
| `/` | Project overview, summary stats, navigation |
| `/pipeline` | Data pipeline visualisation with raw GitHub links to every intermediate file |
| `/mislabels` | Browse the 4,076 cross-validation mislabels with article links |
| `/training` | Per-model training recipes (BART, T5, RL, LoRA) |
| `/eval` | The 7-metric pipeline explained — what / why / score bands / limitations |
| `/dashboard` | 14 models × 7 metrics with filterable bar charts, model-profile radar, strategy breakdown, sortable aggregate table |
| `/explorer` | 2,857-sample browser with filtering, search, side-by-side model comparison |
| `/playground` | Live inference (local mode only) |
| `/human-eval` | Golden eval, multi-classifier audit, heldout set |

### Tech stack

- **Frontend**: Next.js 16 (App Router, React 19, Turbopack), Tailwind
  CSS v4, Recharts. Cohere-inspired design system: 22px radius, white
  canvas, DM Serif Display + DM Sans + JetBrains Mono.
- **Backend**: FastAPI + Pydantic + uvicorn, pandas for data loading.
  Lazy-load BART (HuggingFace transformers, MPS/CUDA/CPU autodetect).
  LLaMA proxied to LMStudio via OpenAI-compatible API.
- **Static export**: `webapp/backend/scripts/export_static.py` dumps
  the entire DataStore to 26 JSON files (~30 MB total).
- **Deployment**: Vercel for static mode; local for live mode.

---

## 11. Reproducibility

### Repository layout

```
Project LLMao/
├── data/
│   ├── raw/                  NHDSD source dataset
│   ├── processed/            Cleaned + paired records
│   │   └── intermediate/     Cross-validation, classifier outputs
│   ├── splits/
│   │   ├── sar_to_non/                    BART-Base, BART-RL train data
│   │   └── sar_to_non_context_enhanced/   BART-CE, BART-CE+RL, LLaMA data
│   └── golden/cleaned/       140 hand-labeled samples × 3 models
│
├── docs/
│   ├── PROJECT.md            ← you are here
│   ├── ARCHITECTURE.md       System overview, domain boundaries
│   ├── DATASET.md            Data sources, schemas, preprocessing
│   ├── METHODS.md            Per-recipe training details
│   ├── EVALUATION.md         7-metric pipeline + human eval methodology
│   ├── EXPERIMENTS.md        Per-model results
│   ├── CORE_BELIEFS.md       Design principles
│   └── POSTER.md             Poster content for the CS4248 final
│
├── results/                  Per-model eval CSVs
│   ├── {model}_results.csv                   Full 2,857-sample eval
│   ├── {model}_results_multi_classifier.csv  3-classifier flip rates
│   └── golden/                               Human-eval analysis
│       ├── summary.csv
│       ├── summary_all_classifiers.csv
│       ├── {model}_merged.csv
│       └── {model}_subtype_analysis.csv
│
├── scripts/
│   ├── train.py              BART seq2seq SFT
│   ├── train_rl.py           REINFORCE + KL on BART checkpoints
│   ├── train_llama.py        LLaMA 3.2 1B LoRA
│   ├── train_llama_context.py  LoRA with article body
│   ├── eval_pipeline.py      7-metric eval pipeline
│   ├── batch_eval.py         Run eval over all 14 models
│   ├── analyze_golden_results.py  Human-eval analysis
│   ├── upload_to_hf.py       Push LLaMA to HuggingFace
│   ├── upload_bart_to_hf.py  Push BART variants to HuggingFace
│   └── data_prep/            Data preprocessing pipeline
│
├── outputs/                  Saved checkpoints + LoRA adapters + GGUF
│
└── webapp/
    ├── backend/              FastAPI inference + static export
    │   ├── app/
    │   │   ├── main.py
    │   │   ├── config.py
    │   │   ├── routers/
    │   │   └── services/
    │   └── scripts/export_static.py
    └── frontend/             Next.js dashboard
        ├── src/app/          Pages (one dir per route)
        ├── src/lib/          API client, constants
        └── public/data/      Pre-exported JSON for static mode
```

T5-family training code lives in
[`camille-readbean/CS4248-project-AY2526S2`](https://github.com/camille-readbean/CS4248-project-AY2526S2).

### Setup

```bash
# Create a virtual environment with the project's Python deps
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# Webapp dependencies
cd webapp/frontend && npm install
```

### Training

```bash
# BART variants (Yang Zhi's pipeline)
python scripts/train.py --model facebook/bart-base --direction sar-to-non
python scripts/train.py --model facebook/bart-base --direction sar-to-non \
    --data_dir data/splits/sar_to_non_context_enhanced  # for BART-CE

# RL refinement
python scripts/train_rl.py \
    --sft_checkpoint outputs/bart-base/sar-to-non/final \
    --classifier_model SeeYangZhi/sarcasm-classifier \
    --epochs 3 --kl_coeff 0.2

# LLaMA LoRA
python scripts/train_llama.py
python scripts/train_llama_context.py

# T5 family (Camille's repo, requires SLURM access)
sbatch scripts/slurm_finetune_t5.sh --model t5-base --datasets joint
sbatch scripts/slurm_finetune_t5.sh --model t5-base \
    --datasets ablation_without_irony ablation_without_overstatement \
               ablation_without_rhetorical_question ablation_without_sarcasm \
               ablation_without_satire ablation_without_understatement
sbatch scripts/slurm_finetune_t5_control.sh t5-base
```

### Evaluation

```bash
# Run the 7-metric pipeline on a single model
python scripts/eval_pipeline.py \
    --input model_outputs_clean/bart_base_rl.csv \
    --output results/bart_base_rl_results.csv \
    --multi_classifier

# Or batch all 14 models at once
python scripts/batch_eval.py --multi_classifier

# Re-analyse the golden set after new annotations
python scripts/analyze_golden_results.py
```

### Webapp

```bash
# Local mode (live inference)
cd webapp/backend && uvicorn app.main:app --reload     # http://localhost:8000
cd webapp/frontend && npm run dev                      # http://localhost:3000

# Static mode (no backend, suitable for Vercel)
python webapp/backend/scripts/export_static.py
NEXT_PUBLIC_USE_STATIC=true npm run build
```

All seeds default to 42. Hyperparameters are saved as
`training_config.json` next to each checkpoint, so any run is
reproducible from its config file.

---

## 12. Lessons Learned

Things we'd do differently if we started again:

1. **Run human evaluation early.** We discovered the multi-classifier
   audit late and spent a long time arguing about which classifier to
   "trust". The right answer was always "none of them; hand-label the
   ground truth". Hand-labeling 140 samples × 2 annotators is ~6 hours
   of work, which would have saved us weeks.

2. **Don't trust LLM annotations as ground truth.** We used StepFun
   for cross-validation and almost overwrote NHDSD's labels with the
   results. The same week, we discovered all three sarcasm classifiers
   are unreliable. **LLMs and small classifiers are equally untrustworthy
   when used outside their training distribution.** Use them as
   audit signals, not as truth.

3. **Train one strong model before running ablations.** We ran the
   6-way ablation in parallel with the BART-RL training, which meant
   the ablation results were sitting around for two weeks before we
   had a baseline to compare them to. Sequential is faster overall
   when you don't know what you're looking for.

4. **Save intermediate checkpoints from RL training.** We discovered
   the BART-RL reward hacking only after the human eval. Having
   checkpoint snapshots from each epoch would have let us see exactly
   when the deletion shortcut was learned.

5. **Pick reward models with care.** The composite reward
   (`α · style + (1−α) · ROUGE-L`) sounds reasonable but neither
   component checks meaning preservation as humans understand it. We
   should have either (a) trained a meaning-aware verifier first or
   (b) used DPO with preference data.

6. **Document the failure modes alongside the successes.** The
   webapp's `/eval` page exists because we discovered, after the
   results were in, that most of our metrics needed asterisks. The
   eval page is now the most-linked page in the entire project.

---

## Glossary

- **Sarcasm style transfer** — the task of generating a non-sarcastic
  rewrite of a sarcastic input that preserves the underlying meaning.
- **Subtype / strategy** — one of six iSarcasm categories: sarcasm,
  irony, satire, overstatement, understatement, rhetorical question.
- **Joint task** — training the model to predict both the sarcastic
  strategy AND the rewrite in a single output string. Used by T5-Joint.
- **Strict success** — the human-eval criterion: sarcasm removed AND
  meaning preserved, both confirmed by both annotators.
- **Flip rate** — fraction of inputs where a sarcasm classifier
  changes its prediction from "sarcastic" to "non-sarcastic" between
  input and output.
- **Multi-classifier audit** — running 3 different sarcasm classifiers
  on the same outputs and reporting the spread, to expose how
  unreliable any single classifier is.
- **Strict success vs flip rate** — strict success is what humans want
  (sarcasm gone, meaning intact); flip rate is what classifiers
  measure (output reads as non-sarcastic in isolation).
- **Reward hacking** — the model finding a shortcut that maximises a
  reward function without achieving the actual goal. See Finding 7.3.
- **Cohen's κ** — inter-rater agreement metric corrected for chance.
  > 0.8 = almost perfect; > 0.6 = substantial; > 0.4 = moderate;
  ~ 0 = chance; < 0 = anti-correlation.
- **NHDSD** — News Headlines Dataset for Sarcasm Detection (Misra
  2019). Our primary data source.
- **iSarcasm** — the taxonomy of six sarcasm subtypes from Abu Farha
  et al. 2022, used as our strategy labels.
- **BART / T5 / LLaMA** — the three model families we train.
- **LoRA** — Low-Rank Adaptation. A parameter-efficient fine-tuning
  method that injects trainable low-rank matrices into the attention
  and MLP projections, leaving the base weights frozen.
- **REINFORCE + KL** — policy-gradient RL with a KL divergence
  penalty against a frozen reference model, used to refine the BART
  SFT checkpoints.
- **GGUF** — GPT-Generated Unified Format. The quantised weight format
  LMStudio uses to serve the merged LLaMA model on consumer hardware.
- **CE (in BART-CE)** — Context-Enhanced. Refers to training data with
  scraped article bodies attached to each pair, **not** cross-entropy.

---

## Appendix A — Full Hyperparameter Tables

### A.1 BART SFT (BART-Base, BART-CE)

| Hyperparameter | Value |
|---|---|
| Trainer | HuggingFace `Seq2SeqTrainer` |
| Backbone | `facebook/bart-base` (140M) |
| Epochs | 5 |
| Early stopping patience | 2 |
| Per-device train batch | 16 |
| Per-device eval batch | 16 |
| Learning rate | 3e-4 |
| LR scheduler | linear with warmup |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Max source length | 128 tokens |
| Max target length | 128 tokens |
| Best metric | BLEU on validation |
| `predict_with_generate` | true |
| Generation max length | 128 |
| Save total limit | 2 |
| Precision | bf16 (CUDA) |
| Seed | 42 |
| Reporting | none (or `--wandb`) |

### A.2 T5 SFT (Joint, Control, Ablations)

| Hyperparameter | Value |
|---|---|
| Trainer | `Seq2SeqTrainer`, `predict_with_generate=True` |
| Backbone (Joint, Control, Ablations) | `google-t5/t5-base` (220M) |
| Backbone (joint legacy) | `google-t5/t5-small` (60M) |
| Epochs | 4 |
| Early stopping | none |
| Per-device train batch | 8 |
| Per-device eval batch | 8 |
| Gradient accumulation | 2 (effective batch 16) |
| Learning rate | 3e-4 |
| LR scheduler | cosine |
| `warmup_ratio` | 0.06 |
| Weight decay | 0.01 |
| Max source length | 1248 tokens |
| Max target length | 1248 tokens |
| Best metric | `eval_loss` |
| Save total limit | 2 |
| Precision | fp16 |
| Seed | 42 |
| Compute | 1× NV GPU, 32G mem, SLURM gpu-long |

### A.3 BART RL (BART-RL, BART-CE+RL)

| Hyperparameter | Value |
|---|---|
| Policy initialisation | SFT BART checkpoint |
| Reference | Same checkpoint, frozen |
| Reward style weight (α) | 0.5 |
| Reward content weight | 0.5 |
| KL coefficient (β) | 0.2 |
| Learning rate | 1e-5 |
| Epochs | 3 |
| Per-device batch | 8 |
| Sampling top-k | 50 |
| Sampling top-p | 0.95 |
| Sampling temperature | 0.8 |
| Reward baseline | EMA, decay 0.9 |
| Gradient clipping | max_norm = 1.0 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Max generation length | 128 tokens |
| Seed | 42 |

### A.4 LLaMA LoRA (LLaMA 3.2 1B base + context)

| Hyperparameter | Value (base) | Value (context) |
|---|---|---|
| Backbone | `meta-llama/Llama-3.2-1B-Instruct` | same |
| LoRA rank `r` | 16 | 16 |
| LoRA alpha | 32 | 32 |
| LoRA dropout | 0.05 | 0.05 |
| Target modules | q, k, v, o, gate, up, down | same |
| Trainable params | ~6M of 1.24B (0.5%) | same |
| Learning rate | 2e-4 | 2e-4 |
| Per-device batch | 8 | 4 |
| Gradient accumulation | 2 (eff. 16) | 4 (eff. 16) |
| Epochs | 3 | 3 |
| Max sequence length | 256 tokens | 1024 tokens |
| LR scheduler | cosine | cosine |
| Warmup ratio | 0.05 | 0.05 |
| Weight decay | 0.01 | 0.01 |
| Precision | bf16 | bf16 |
| Gradient checkpointing | true | true |
| Loss masking | prompt → −100, response only | same |
| Seed | 42 | 42 |

---

## Appendix B — LLM Annotation Prompts

### B.1 Pair generation prompt (sarcastic → non-sarcastic)

> You are a writing assistant. Given a sarcastic news headline, generate
> a non-sarcastic, neutral version that preserves the underlying claim.
> Respond with only the rewritten headline, no explanation.
>
> Examples:
> Sarcastic: "Area Man Passionate Defender Of What He Imagines Constitution To Be"
> Non-sarcastic: "Local man strongly defends his personal interpretation of the Constitution"
>
> Sarcastic: "Nation's Dog Owners Resolve To Be More Forgiving After Learning How Hard It Is To Be A Dog"
> Non-sarcastic: "Dog owners express greater empathy for their pets after considering their daily lives"
>
> Sarcastic: "Study Finds Every Style Of Parenting Produces Disturbed, Miserable Adults"
> Non-sarcastic: "Study finds correlations between various parenting styles and adult mental health outcomes"
>
> Now rewrite this:
> Sarcastic: "{INPUT_HEADLINE}"
> Non-sarcastic:

### B.2 Strategy classification prompt

> You are a linguist labelling sarcasm strategies. Given a sarcastic
> headline, classify which of these six mechanisms it uses:
>
> - sarcasm: contradicts state of affairs, critical
> - irony: contradicts state of affairs, not critical
> - satire: mimics a serious genre with mockery
> - overstatement: obviously exaggerated
> - understatement: neutral words for extreme situations
> - rhetorical_question: question whose answer is implicit
>
> Reply with only the strategy name.
>
> Headline: "{HEADLINE}"
> Strategy:

### B.3 LLaMA system prompt (training and inference)

> You are a writing assistant. Rewrite sarcastic news headlines as
> neutral, factual equivalents that preserve the core meaning without
> irony or mockery. Respond with only the rewritten headline, no
> explanation.

### B.4 LLM-as-judge prompt (Gemini 2.5 Flash)

> You are evaluating a sarcasm style transfer model. The input is a
> sarcastic news headline; the output is the model's attempt to
> rewrite it as a neutral, factual equivalent.
>
> Rate the output on three dimensions, each 1–5:
>
> 1. **sarcasm_removed** (1 = still very sarcastic, 5 = completely
>    neutral)
> 2. **meaning_preserved** (1 = entirely different meaning, 5 = exact
>    same underlying claim)
> 3. **fluency** (1 = ungrammatical, 5 = natural English)
>
> Reply in JSON: `{"sarcasm_removed": int, "meaning_preserved": int, "fluency": int}`
>
> Input: "{INPUT}"
> Output: "{OUTPUT}"

---

## References

- Abu Farha, I., Oprea, S. V., Wilson, S., & Magdy, W. (2022).
  **SemEval-2022 Task 6: iSarcasmEval, Intended Sarcasm Detection in
  English and Arabic.** *SemEval-2022*.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S.,
  Wang, L., & Chen, W. (2021). **LoRA: Low-Rank Adaptation of Large
  Language Models.** *arXiv:2106.09685*.
- Landis, J. R., & Koch, G. G. (1977). **The measurement of observer
  agreement for categorical data.** *Biometrics*, 33(1), 159–174.
- Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy,
  O., Stoyanov, V., & Zettlemoyer, L. (2020). **BART: Denoising
  Sequence-to-Sequence Pre-training for Natural Language Generation,
  Translation, and Comprehension.** *ACL 2020*.
- Misra, R. (2019). **News Headlines Dataset for Sarcasm Detection.**
  *Kaggle*.
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., &
  Finn, C. (2023). **Direct Preference Optimization: Your Language
  Model is Secretly a Reward Model.** *NeurIPS 2023*.
- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena,
  M., Zhou, Y., Li, W., & Liu, P. J. (2020). **Exploring the Limits of
  Transfer Learning with a Unified Text-to-Text Transformer.**
  *JMLR* 21.
- Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin,
  C., Liang, P., & Hashimoto, T. B. (2023). **Stanford Alpaca: An
  Instruction-following LLaMA model.** *Stanford CRFM*.
- Touvron, H., Lavril, T., Izacard, G., et al. (2023). **LLaMA 2: Open
  Foundation and Fine-Tuned Chat Models.** *arXiv:2307.09288*.
- Williams, R. J. (1992). **Simple Statistical Gradient-Following
  Algorithms for Connectionist Reinforcement Learning.** *Machine
  Learning*, 8(3-4), 229–256.
- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang,
  Y., et al. (2023). **Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena.** *NeurIPS 2023*.
- Zhu, K., Hu, J., Liu, Y., Ouyang, J., Wu, F., Yan, R., & Wang, X.
  (2025). **ViSP: Visual Sarcasm Generation with PPO Reinforcement
  Learning.** *arXiv:2507.09482*.

---

_Last updated: 2026-04-14_
