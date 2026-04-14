# Methods

> Training recipes for the 14 models we evaluate.

## Task Definition

**Input**: a sarcastic news headline.
**Output**: a non-sarcastic equivalent that preserves the underlying claim.

```
Sarcastic:     "Area Man Passionate Defender Of What He Imagines Constitution To Be"
Non-sarcastic: "Local man strongly defends his personal interpretation of the Constitution"
```

Six subtype labels from the iSarcasm taxonomy mediate the rewrite:
`sarcasm`, `irony`, `satire`, `overstatement`, `understatement`,
`rhetorical_question`. Strategy-aware models are explicitly conditioned
on (or asked to predict) the subtype before generating.

## Data

| Split | Used by | Records | Source |
|---|---|---|---|
| `data/splits/sar_to_non/` | BART-Base, BART-RL | 10,868 / 1,356 / 1,364 | `scripts/data_prep/create_train_val_test_splits.py` |
| `data/splits/sar_to_non_context_enhanced/` | BART-CE, BART-CE+RL, both LLaMA variants | 8,258 / 1,029 | Subset with scraped article body |
| `data/joint_and_ablate_prepared/` | T5-Joint, T5-Control, 6 ablations | 80/10/10 stratified | `prepare_t5_datasets.py` (Camille's repo) |

See [DATASET.md](DATASET.md) for the full preprocessing pipeline.

## Training Recipes

Four distinct recipes, 14 models total. The `/training` page in the
webapp renders this same information interactively with per-model cards.

### Recipe 1 — BART supervised fine-tuning (Yang Zhi, this repo)

**Models**: BART-Base, BART-CE
**Script**: [`scripts/train.py`](../scripts/train.py)
**Backbone**: `facebook/bart-base` (140M)

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
| Input format | Raw sarcastic headline (no prefix; BART is pretrained with denoising) |

BART-Base trains on `sar_to_non/`; BART-CE trains on the
context-enhanced split where each pair has a scraped article body
attached. Same recipe, different data.

### Recipe 2 — T5 supervised fine-tuning (Camille's separate repo)

**Models**: T5-Joint, T5-Control, six ablations
**Script**: [`finetune_T5.py`](https://github.com/camille-readbean/CS4248-project-AY2526S2/blob/main/scripts/finetune_T5.py)
**Orchestration**: SLURM via [`slurm_finetune_t5.sh`](https://github.com/camille-readbean/CS4248-project-AY2526S2/blob/main/scripts/slurm_finetune_t5.sh)
**Backbone**: `google-t5/t5-base` (220M); the older `joint` variant is t5-small (60M)

| Hyperparameter | Value |
|---|---|
| Trainer | HuggingFace `Seq2SeqTrainer`, `predict_with_generate=True` |
| Epochs | 4 (no early stopping) |
| Per-device batch | 8 |
| Grad accumulation | 2 (effective batch 16) |
| Learning rate | 3e-4 |
| Scheduler | Cosine, warmup_ratio 0.06 |
| Weight decay | 0.01 |
| Max source/target length | 1248 tokens |
| Best metric | `eval_loss` |
| Precision | fp16 |
| Compute | 1× NV GPU, 32G mem, SLURM `gpu-long`, 5h limit |

The three task variants use different input/target formats:

| Variant | Input prefix | Target |
|---|---|---|
| T5-Joint | `"rewrite to non-sarcastic and predict strategy: "` | `"strategy: {strategy} rewrite: {non_sarcastic}"` |
| T5-Control | `"rewrite to non-sarcastic: "` | `{non_sarcastic}` (plain) |
| Ablations | `"rewrite to non-sarcastic: "` | `{non_sarcastic}` (plain) |

**T5-Joint is the best model overall on human evaluation** (43.6% strict
success vs T5-Control's 39.3%). Its only structural advantage is the
strategy-prediction prefix, which forces task decomposition before
generation — see `docs/EVALUATION.md` for the analysis.

**Ablation construction** (`prepare_t5_datasets.py`): one of the six
subtypes is dropped from train+val pools; remaining pools are stratified-
downsampled to the minimum-across-drops to keep effective dataset size
constant. Test set is the full held-out split, shared across all six.

### Recipe 3 — Reinforcement learning (BART variants)

**Models**: BART-RL, BART-CE+RL
**Script**: [`scripts/train_rl.py`](../scripts/train_rl.py)
**Initialisation**: SFT BART checkpoint as both policy and frozen reference

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

**Loss formulation**:

```
r       = α · (1 − P_sarcastic(output)) + (1 − α) · ROUGE-L(output, ref)
L_rl    = −(r − baseline) · Σ log π_θ(y | x)
L_total = L_rl + β · KL(π_θ || π_ref)
```

| Hyperparameter | Value |
|---|---|
| α (style weight) | 0.5 |
| β (KL coeff) | 0.2 |
| Learning rate | 1e-5 (much lower than SFT — drift risk) |
| Epochs | 3 |
| Baseline | EMA of batch reward, decay 0.9 |
| Gradient clipping | max_norm 1.0 |
| Sampling | top-k=50, top-p=0.95, temp=0.8 |

**Pure style reward saturates** because the SFT outputs already classify
as ~1.0 non-sarcastic on the reward model. The ROUGE-L term provides
content signal — without it the model learns to delete tokens to satisfy
the classifier (see Recipe 3 failure mode below).

**Known failure mode — reward hacking**: BART-RL achieves the highest
classifier flip rates but human eval shows it has a 40.7% meaning-change
rate (more than double T5-Joint's 16.4%). The classifier reward gradient
is satisfied by deleting sarcastic tokens, not by genuine rewriting. This
is the central cautionary finding of the project.

### Recipe 4 — LoRA instruction tuning (LLaMA)

**Models**: LLaMA 3.2 1B, LLaMA 3.2 1B (context)
**Scripts**: [`scripts/train_llama.py`](../scripts/train_llama.py),
[`scripts/train_llama_context.py`](../scripts/train_llama_context.py)
**Backbone**: `meta-llama/Llama-3.2-1B-Instruct` (1.24B)

LLaMA is decoder-only, so the recipe diverges from the seq2seq pipeline.
The chat template wraps a fixed system prompt and the user message; loss
is masked to the assistant response only (prompt tokens set to −100 in
labels).

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
| Target modules | `q, k, v, o, gate, up, down` (all attention + MLP projections) |
| Trainable params | ~6M of 1.24B (0.5%) |
| Learning rate | 2e-4 |
| Batch × grad accum | 8 × 2 = 16 effective |
| Epochs | 3 |
| Max length (base) | 256 tokens |
| Max length (context) | 1024 tokens (to fit article body, batch drops to 4×4) |
| Scheduler | Cosine, 5% warmup |
| Precision | bf16 + gradient checkpointing |

After training the LoRA adapter is merged back into the base model and
exported to GGUF for LMStudio so the webapp's `/playground` page can
serve it on consumer hardware.

## Inference

The webapp `/playground` page exposes two live models behind a FastAPI
backend:

- **BART-CE+RL**: HuggingFace transformers, MPS / CUDA / CPU autodetect
- **LLaMA 3.2 1B**: served by LMStudio at `localhost:1234/v1`

T5 models are seq2seq and could be served directly from HuggingFace, but
LMStudio currently only handles decoder-only checkpoints. The static
Vercel deployment disables live inference and shows a hosted-version
banner explaining how to run it locally.

## Evaluation

Every model is evaluated on the full 2,857-sample test split with the
7-metric pipeline (`scripts/eval_pipeline.py --multi_classifier`):

1. Sarcasm flip rate (× 3 classifiers)
2. Semantic similarity (sentence-transformers/all-MiniLM-L6-v2)
3. Perplexity (GPT-2 reference)
4. BLEU vs input
5. Edit distance (word-level Levenshtein, normalised)
6. LLM-as-judge (Gemini 2.5 Flash, sampled subset)
7. Paraphrase score (similarity × (1 − BLEU))

Three of the 14 models (T5-Joint, T5-Control, BART-RL) additionally have
hand-labeled human evaluation on 140 stratified samples by 2 annotators.
See [EVALUATION.md](EVALUATION.md) for the full methodology and findings.

## Implementation Notes

- All training runs save `training_config.json` next to the checkpoint
- Seeds default to 42 and are logged in the saved config
- Weights & Biases is optional via `--wandb` on every script
- LLaMA exports both LoRA adapter (~50MB) and merged + GGUF (~2GB)

---

_Last updated: 2026-04-14_
