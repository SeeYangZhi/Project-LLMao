# Project LLMao

> **Decoding Sarcasm with LLMao: Lightweight Language Models with
> Aspect-aware Objectives**

CS4248 · Team 14 · National University of Singapore · AY2025/26 S2

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
   T5-Control and BART-RL on the same 140-sample set. Same data, same
   recipe — only the joint task formulation changes — and meaning
   preservation improves from 75% to 84%.
3. **Reinforcement learning with classifier rewards is reward hacking.**
   BART-RL achieves the highest classifier flip rates of any model but
   human evaluation reveals a 40.7% meaning-change rate. The composite
   `style + ROUGE-L` reward is satisfied by deleting sarcastic tokens;
   the KL penalty against the SFT reference is not enough to stop it.

The full results, methodology, and an interactive dashboard live in the
[webapp](../webapp/) and the per-page documentation linked at the
bottom.

---

## 1. Motivation

### Why sarcasm style transfer?

Sarcasm is a long-standing failure mode for NLP systems. Sentiment
analysers misread "Great job on the update, everything is broken now"
as positive. Content moderation pipelines miss the intended meaning of
satirical headlines. Most existing work treats sarcasm detection as a
black-box prediction task — the model outputs a label but offers no
insight into *what makes the headline sarcastic* or *what the
underlying claim actually is*.

We reframe the problem as **style transfer**: rather than classifying
sarcasm, *remove* it. Given a sarcastic headline, generate a neutral,
factual equivalent that preserves the underlying claim. This forces
the model to demonstrate understanding (you can't rewrite what you
don't comprehend) and produces a directly useful artefact (a
sentiment-tagger-friendly version of the same news event).

### Why small models?

We deliberately scope the work to **small language models** (60M–1.3B
parameters) rather than large frontier LLMs. Three reasons:

- **Inspectable**: a fine-tuned T5-base allows ablation of control codes,
  reproducible training, and per-sample analysis. Frontier LLMs are
  black boxes.
- **Practical**: a fine-tuned T5-base runs inference in ~10 ms on a
  single GPU. An LLM API call costs per token and takes 1–2 s. For any
  deployable application the small model is necessary.
- **Controllable**: our strategy-aware models respond to explicit
  strategy tokens for deterministic, structured output. A frontier LLM
  with a prompt does not offer this guarantee.

The frontier LLM still appears in the project — but as a **synthetic
data annotator**, not as the model that solves the task. This separates
the dataset construction question (how do we get supervision that
doesn't exist in the wild) from the research question (can small,
efficient models learn from that supervision).

---

## 2. Data

### Source

**News Headlines Dataset for Sarcasm Detection (NHDSD)**, 28,619
headlines: 13,634 sarcastic from TheOnion + 14,985 non-sarcastic from
HuffPost. We use NHDSD because:

- Headlines are professionally written (no spelling noise)
- They're self-contained (no reply/thread context required)
- TheOnion's sole purpose is sarcastic news, so labels are reliable

### Data Quality Audit

NHDSD's labels turn out to be noisier than its reputation suggests.
We relabeled every headline with **StepFun 3.5 Flash** and found 80.19%
agreement with the original labels. Of the 5,644 disagreements, **4,076
were confirmed as suspected mislabels** by a second LLM (Nemotron 3
Nano). The webapp's [`/mislabels`](../webapp/frontend/src/app/mislabels/page.tsx)
page surfaces all of these with article links so a human can judge.

We did **not** silently overwrite the labels — the main pipeline still
trains on raw NHDSD because the cross-validation is itself an
audit-only signal that uses two LLMs whose own reliability we
question. Treating it as ground truth would be the same mistake we
diagnose later in the classifier audit.

### Synthetic Parallel Corpus

No large-scale sarcastic↔non-sarcastic paired dataset exists. We
construct one with StepFun 3.5 Flash via OpenRouter:

- **Sarcastic → Non-sarcastic**: 13,588 pairs
- **Non-sarcastic → Sarcastic**: 14,948 pairs
- **Total**: 28,536 raw pairs

For every non→sarcastic pair we additionally generate **5 strategy
variants** (one per missing subtype from the 6 iSarcasm categories),
producing 89,688 strategy-annotated records with perfect class balance
(14,948 per subtype). This augmented corpus is the training data for
the joint-task and ablation models.

### Train/Val/Test Splits

Two splits power the downstream experiments:

| Split | Used by | Train / Val / Test |
|---|---|---|
| `data/splits/sar_to_non/` | BART-Base, BART-RL | 10,868 / 1,356 / 1,364 |
| `data/splits/sar_to_non_context_enhanced/` | BART-CE, BART-CE+RL, both LLaMA variants | 8,258 / 1,029 / — |
| `data/joint_and_ablate_prepared/` | T5-Joint, T5-Control, 6 ablations | 80/10/10 stratified by subtype |

Source-level splits prevent leakage (all six strategy variants of a
given source headline land in the same split).

A separate **140-sample golden set** is hand-labeled by 2 annotators
for the human evaluation — stratified across the six subtypes,
covering T5-Joint, T5-Control, and BART-RL.

See [`docs/DATASET.md`](DATASET.md) for the full preprocessing
pipeline.

---

## 3. Models

We train **14 models** in total across **four recipes**. Two
independent pipelines feed into the same downstream evaluation:

### Recipe 1 — BART supervised fine-tuning (Yang Zhi, this repo)

Two BART variants distinguished only by training data:

- **BART-Base**: SFT on the main `sar_to_non` split (10,868 pairs)
- **BART-CE**: SFT on the context-enhanced split (8,258 pairs with
  scraped article body)

Same recipe: HuggingFace `Seq2SeqTrainer`, 5 epochs with early stop on
val BLEU, batch 16, LR 3e-4, max length 128, bf16. BART takes the raw
headline as input — no task prefix, since BART is pretrained with its
own denoising objective.

### Recipe 2 — T5 supervised fine-tuning (Camille's separate repo)

The T5 family lives in
[`camille-readbean/CS4248-project-AY2526S2`](https://github.com/camille-readbean/CS4248-project-AY2526S2)
with its own pipeline (`finetune_T5.py` + SLURM orchestration) and a
different training recipe than the BART side: 4 epochs, per-device
batch 8, grad accumulation 2 (effective 16), LR 3e-4, max source/target
1248 tokens, fp16, eval_loss as the best metric.

Three task variants share that recipe:

- **T5-Joint**: target format `"strategy: {strategy} rewrite:
  {non_sarcastic}"` — forces the model to predict the strategy before
  rewriting
- **T5-Control**: target is the plain rewrite, no strategy token —
  isolates the contribution of the joint objective
- **T5-Joint (small)**: same recipe, t5-small backbone instead of
  t5-base — kept to show that T5-Joint's edge is the joint task, not
  the larger backbone

Plus **6 ablation models** that each drop one sarcasm subtype from
train+val. Pools are stratified-downsampled to the minimum across all
six drops to keep effective dataset size constant; the test set is the
full split, shared across all six.

### Recipe 3 — Reinforcement learning (BART variants)

Two RL models built on top of the BART SFT checkpoints:

- **BART-RL**: starts from BART-Base SFT
- **BART-CE+RL**: starts from BART-CE SFT

The recipe is REINFORCE with a KL penalty against the frozen SFT
reference. The composite reward blends a sarcasm classifier signal
with content preservation:

```
r       = α · (1 − P_sarcastic(output)) + (1 − α) · ROUGE-L(output, ref),  α = 0.5
L_rl    = −(r − baseline) · Σ log π_θ(y | x)
L_total = L_rl + β · KL(π_θ || π_ref),  β = 0.2
```

Pure style reward saturates instantly because the SFT outputs already
classify as ~1.0 non-sarcastic — the ROUGE-L term is meant to keep the
model from collapsing to "delete everything". **It doesn't quite
work**: see Finding 3 below.

### Recipe 4 — LoRA instruction tuning (LLaMA)

Two LLaMA variants on `meta-llama/Llama-3.2-1B-Instruct`:

- **LLaMA 3.2 1B**: headline-only prompt
- **LLaMA 3.2 1B (context)**: prompt includes the scraped article body

PEFT LoRA (r=16, α=32, dropout 0.05) on all 7 attention + MLP
projections — only ~6M of 1.24B parameters are trained (0.5%). LR
2e-4, effective batch 16, 3 epochs, cosine schedule, 5% warmup, bf16
+ gradient checkpointing. The system prompt and user message are
encoded with the Llama 3 chat template; loss is masked to the
assistant response only. After training the LoRA adapter is merged
back into the base model and exported to GGUF for LMStudio.

The full per-recipe table with hyperparameters is on the webapp's
[`/training`](../webapp/frontend/src/app/training/page.tsx) page and
in [`docs/METHODS.md`](METHODS.md).

---

## 4. Evaluation

### 7-Metric Automated Pipeline

`scripts/eval_pipeline.py --multi_classifier` produces seven complementary
metrics for every output:

1. **Sarcasm flip rate** × **3 classifiers** (RoBERTa-Twitter,
   Bert-Kaggle, RoBERTa-News)
2. **Semantic similarity** (sentence-transformers/all-MiniLM-L6-v2)
3. **Perplexity** (GPT-2 reference LM)
4. **BLEU vs input** (detects copying vs genuine rewriting)
5. **Edit distance** (word-level Levenshtein, normalised)
6. **LLM-as-judge** (Gemini 2.5 Flash, sampled subset, scores
   sarcasm_removed / meaning_preserved / fluency 1–5)
7. **Paraphrase score** = `similarity × (1 − BLEU)` — combines
   meaning preservation and rewriting depth into a single number that
   neither metric captures alone

Every model in the project is evaluated on the full 2,857-sample test
split. The webapp's [`/eval`](../webapp/frontend/src/app/eval/page.tsx)
page documents what each metric measures, the score interpretation
bands, and the limitations.

### Human Evaluation

For three models (T5-Joint, T5-Control, BART-RL) we hand-label **140
stratified samples × 2 independent annotators** on two binary
questions:

- **`sarcasm_removed`** — does the output read as non-sarcastic?
- **`meaning_change`** — has the underlying claim been altered?

Inter-annotator κ across the three models is 0.839, 0.883, and 0.884
— excellent agreement, so the labels are reliable ground truth.
**Strict success** is defined as `sarcasm_removed AND NOT
meaning_change`.

---

## 5. Findings

### Finding 1: Automated flip rate is not a valid primary metric

Three sarcasm classifiers from different domains, three completely
different stories about the same outputs:

| Classifier | Avg κ vs human (across 3 models) |
|---|---|
| RoBERTa-Twitter | +0.019 |
| Bert-Kaggle | **−0.075** |
| RoBERTa-News | +0.104 |
| **Human inter-annotator** | **0.84+** |

**4 of 9 model×classifier cells show negative κ** — the classifier
anti-correlates with humans. The best κ across the entire matrix is
+0.18. By comparison, two random untrained humans would average around
0.

The classifiers were trained for *sarcasm detection* ("is this single
text sarcastic?") but we use them for *removal verification* ("did the
rewrite become non-sarcastic relative to the input?"). They never see
the input/output pair together, so they can't detect transformation —
only static lexical surface. **Picking any one classifier as "the"
flip rate would have given us a confidently wrong story about which
model is best.**

### Finding 2: T5-Joint is the best model overall

| Metric | T5-Joint | T5-Control | BART-RL |
|---|---|---|---|
| **Strict success** | **43.6%** | 39.3% | 34.3% |
| Human flip rate | 54.3% | 54.3% | 52.9% |
| Meaning change rate | **16.4%** | 25.0% | 40.7% |

T5-Joint wins because the strategy-prediction prefix forces the model
to identify *what's sarcastic* before generating the rewrite. The
joint output format is:

```
strategy: irony rewrite: Local man strongly defends his interpretation of the Constitution
```

T5-Control trains on the same data with the same hyperparameters but
emits only the rewrite. Adding the strategy token to the target
improves meaning preservation from 75% (control) to 84% (joint) — a
single-knob win on the property humans care about most.

### Finding 3: BART-RL is reward hacking

BART-RL achieves the highest classifier flip rates of any model in
the project but human eval reveals a **40.7% meaning-change rate** —
more than double T5-Joint's 16.4%. The composite reward is satisfied
by deleting sarcastic tokens while keeping enough word overlap to
preserve ROUGE-L:

```
Input:  "Area Man Proud Of Completely Average Achievement"
BART-RL: "Man achieves something."
```

The classifier is happy. The annotator marks the meaning destroyed.
The KL penalty against the SFT reference (β = 0.2) is not enough to
stop it. **This is the central cautionary finding of the project**:
optimising a reward you can't trust produces a model you can't trust.

### Finding 4: LLaMA and BART-CE rewrite too aggressively

| Model | Similarity | Edit Distance | LLM Meaning |
|---|---|---|---|
| LLaMA 3.2 1B | 0.66 | 0.95 | 3.34 / 5 |
| BART-CE | 0.64 | 0.92 | 3.52 / 5 |
| BART-CE+RL | 0.61 | 0.93 | 2.90 / 5 |

These models edit ~95% of the input and retain ~1% n-gram overlap.
That's *generation* with the input as a prompt, not *style transfer*.
LLaMA hallucinates plausible-but-fabricated facts; BART-CE+RL drifts
even further. Larger generative capacity helps comprehension but the
model forgets to preserve the original.

### Finding 5: Subtype ablations are interchangeable

Six retrains of the T5 control recipe with one subtype dropped each.
All six cluster within **0.005 similarity** of each other on the test
set. No single subtype is load-bearing.

**Interpretation**: sarcasm subtypes share underlying mechanisms
(hyperbole, contradiction, absurdity) and the model learns generic
patterns that transfer across categories. A positive generalisation
finding, and an explanation for why oversampling one subtype doesn't
help.

### Finding 6: Different subtypes fail for different reasons

| Subtype | Classifier Miss Rate | Why |
|---|---|---|
| satire | 80% | Mimics legitimate news format |
| rhetorical_question | 71% | Sarcasm in pragmatics, not lexicon |
| irony | 67% | Contradiction is contextual, no surface markers |
| understatement | 50% | Requires world knowledge of "appropriate" response |
| sarcasm (generic) | 44% | Has detectable lexical patterns |
| overstatement | 100%* | **Model failure** — only 28.6% human flip rate |

†% of human-labeled flips that the classifier failed to detect.
*Overstatement is anomalous: 100% miss rate isn't classifier failure
but model failure — only 28.6% of overstatement headlines are
successfully de-sarcasm'd by humans. The model can't remove what
defines the headline.

The full per-subtype breakdown by classifier is on the webapp's
[`/human-eval`](../webapp/frontend/src/app/human-eval/page.tsx) page.

---

## 6. Webapp

A live demo and interactive dashboard:

- **Local mode**: FastAPI backend (T5/BART/LLaMA inference via
  HuggingFace transformers + LMStudio for the LLaMA path)
- **Static mode**: pre-exported JSON consumed by a Next.js frontend
  deployed to Vercel; live inference is disabled but every chart and
  table works

Pages:

| Route | What's there |
|---|---|
| `/` | Project overview, summary stats, navigation |
| `/pipeline` | Data pipeline visualisation with raw GitHub links to every intermediate file |
| `/mislabels` | Browse the 4,076 cross-validation mislabels with article links |
| `/training` | Per-model training recipes (BART, T5, RL, LoRA) |
| `/eval` | The 7-metric pipeline explained — what / why / score bands / limitations |
| `/dashboard` | 14 models × 7 metrics with filterable bar charts, model-profile radar, strategy breakdown, sortable aggregate table |
| `/explorer` | 2,857-sample browser with filtering, search, and side-by-side model comparison |
| `/playground` | Live inference (local mode only) |
| `/human-eval` | Golden eval, multi-classifier audit, heldout set |

Build/run instructions are in `webapp/frontend/README.md` and
`webapp/backend/README.md`.

---

## 7. Repository Layout

```
Project LLMao/
├── data/
│   ├── raw/                  NHDSD source dataset
│   ├── processed/            Cleaned + paired records
│   ├── splits/               sar_to_non / sar_to_non_context_enhanced
│   └── golden/               140 hand-labeled samples × 3 models
├── docs/
│   ├── PROJECT.md            ← you are here
│   ├── ARCHITECTURE.md       System overview, domain boundaries
│   ├── DATASET.md            Data sources, schemas, preprocessing
│   ├── METHODS.md            Per-recipe training details
│   ├── EVALUATION.md         7-metric pipeline + human eval methodology
│   ├── EXPERIMENTS.md        Per-model results
│   ├── CORE_BELIEFS.md       Design principles
│   └── POSTER.md             Poster content for the CS4248 final
├── results/                  Per-model eval CSVs + golden eval CSVs
├── scripts/
│   ├── train.py              BART seq2seq SFT (Yang Zhi)
│   ├── train_rl.py           REINFORCE + KL on BART checkpoints
│   ├── train_llama.py        LLaMA 3.2 1B LoRA
│   ├── train_llama_context.py  LoRA with article body
│   ├── eval_pipeline.py      7-metric eval pipeline
│   ├── batch_eval.py         Run eval over all 14 models
│   └── analyze_golden_results.py  Human-eval analysis
├── outputs/                  Saved checkpoints + LoRA adapters + GGUF
└── webapp/
    ├── backend/              FastAPI inference + static export
    └── frontend/             Next.js dashboard
```

T5-family training code lives in
[`camille-readbean/CS4248-project-AY2526S2`](https://github.com/camille-readbean/CS4248-project-AY2526S2).

---

## 8. Reproducing the Results

```bash
# BART + LLaMA (this repo)
python scripts/train.py --model facebook/bart-base --direction sar-to-non
python scripts/train_rl.py --sft_checkpoint outputs/bart-base/sar-to-non/final
python scripts/train_llama.py
python scripts/train_llama_context.py

# T5 family (Camille's repo)
sbatch scripts/slurm_finetune_t5.sh --model t5-base
sbatch scripts/slurm_finetune_t5_control.sh t5-base

# Evaluation (this repo)
python scripts/batch_eval.py --multi_classifier
python scripts/analyze_golden_results.py

# Webapp (local mode)
cd webapp/backend && uvicorn app.main:app --reload    # localhost:8000
cd webapp/frontend && npm run dev                     # localhost:3000
```

All seeds default to 42. Hyperparameters are saved as
`training_config.json` next to each checkpoint, so any run is
reproducible from its config file.

---

## 9. Team & Acknowledgements

**Team 14, CS4248 (NUS, AY2025/26 S2)**

**Mentor**: Xiao Yu

We thank the NHDSD authors (Misra 2019), the iSarcasmEval taxonomy
(Abu Farha et al. 2022), the helinivan / cardiffnlp / jkhan447 sarcasm
classifier checkpoints on HuggingFace, the StepFun and Nemotron teams
for free OpenRouter access during dataset construction, and the
NUS SLURM cluster.

---

## References

- Misra, R. (2019). News Headlines Dataset for Sarcasm Detection.
  Kaggle.
- Abu Farha, I. et al. (2022). SemEval-2022 Task 6: iSarcasmEval —
  Intended Sarcasm Detection in English and Arabic.
- Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning
  with a Unified Text-to-Text Transformer (T5).
- Lewis, M. et al. (2020). BART: Denoising Sequence-to-Sequence
  Pre-training for Natural Language Generation.
- Touvron, H. et al. (2024). The Llama 3 Herd of Models.
- Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language
  Models.
- Williams, R. J. (1992). Simple Statistical Gradient-Following
  Algorithms for Connectionist Reinforcement Learning (REINFORCE).
- Zhu, K. et al. (2025). ViSP: Visual Sarcasm Generation with PPO
  Reinforcement Learning. arXiv:2507.09482.
- Taori, R. et al. (2023). Alpaca: A Strong, Replicable
  Instruction-Following Model.

---

_Last updated: 2026-04-14_
