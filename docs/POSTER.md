# Poster Content: Project LLMao

> Sarcasm Style Transfer — De-sarcasm via Strategy-Aware Fine-Tuning of Small Language Models

**CS4248 Team 14**: 
**Mentor**: Xiao Yu

---

## Abstract

Sarcasm poses a significant challenge for NLP systems — sentiment analyzers misread sarcastic text as positive, and content moderation pipelines fail to capture the intended meaning. We tackle **sarcasm style transfer**: given a sarcastic headline, generate a non-sarcastic equivalent that preserves the underlying meaning. Since no large-scale paired dataset exists for this task, we construct a synthetic parallel corpus of 89,688 strategy-annotated pairs using LLM-based generation (StepFun Step-3.5 Flash) with cross-validation (Nemotron). We fine-tune three small models — **T5-base**, **BART-base**, and **GPT-2** — on 13,588 sarcastic-to-non-sarcastic pairs using supervised fine-tuning (SFT), and find that small models (124M–250M parameters) learn only surface-level paraphrasing. To address this, we apply **reinforcement learning** (REINFORCE with KL penalty) using our sarcasm classifier (Macro F1: 0.938) as the reward signal, following the ViSP framework. We evaluate with BLEU, METEOR, ROUGE-L, and classifier-based style accuracy, demonstrating that RL refinement can push small models beyond surface rewriting toward genuine style transfer.

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

### Model Architectures

**T5-base** (220M params, seq2seq)
- Input: `desarcasm: {sarcastic headline}`
- Output: non-sarcastic equivalent
- Trainer: `Seq2SeqTrainer` with `predict_with_generate`

**BART-base** (139M params, denoising seq2seq)
- Input: `{sarcastic headline}` (no task prefix — BART is not pretrained with prefixes)
- Output: non-sarcastic equivalent
- Trainer: `Seq2SeqTrainer`

**GPT-2** (124M params, causal LM)
- Input: `{sarcastic headline} → {target}`
- Loss masked on input tokens (only train on target generation)
- Custom data collator for variable-length padding

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 3e-4 |
| Batch size | 16 |
| Max epochs | 5 |
| Max sequence length | 128 |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Early stopping patience | 2 |
| Metric (seq2seq) | BLEU |
| Metric (causal) | eval_loss |

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

**Reward**: Composite score — `r = α · (1 - P(sarcastic)) + (1-α) · ROUGE-L(output, reference)`
- Style reward from DistilBERT classifier (Macro F1: 0.938): high reward = output reads as non-sarcastic
- Content reward from ROUGE-L: penalizes outputs that lose meaning
- Connects classification (Part 1) and generation (Part 2) of our project

**KL penalty**: Prevents reward hacking — model can't drift too far from the coherent SFT baseline

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

### 3.1 Automatic Metrics (Sar-to-Non Test Set)

<!-- TODO: Fill in exact numbers from Colab runs -->

| Model | BLEU | METEOR | ROUGE-L | Style Acc. |
|-------|------|--------|---------|------------|
| T5-base (SFT) | 0.XX | 0.XX | 0.XX | 0.XX |
| BART-base (SFT) | 0.2934 | 0.XX | 0.XX | 0.XX |
| GPT-2 (SFT) | 0.1822 | 0.XX | 0.XX | 0.XX |
| **BART-base (SFT + RL)** | **0.XX** | **0.XX** | **0.XX** | **0.XX** |

*Style Acc. = fraction of outputs classified as non-sarcastic by the reward model*

### 3.2 Sample Outputs — Onion Headlines (SFT vs RL)

| Input (sarcastic) | BART (SFT) | BART (SFT + RL) |
|--------------------|------------|------------------|
| "Panicked Dodgers Owner Has No Idea How He's Going To Come Up With $414 Million Payroll" | dodgers owner unsure how to raise $414 million | The panicked Dodgers owner is unsure how he will come up with $414 million payroll. |
| "Hospital Decides Cancer-Sniffing Leopard More Trouble Than It's Worth" | hospital warns cancer-sniffing leopard may cause more trouble than it's worth | Hospital believes a cancer-sniffing leopard is causing more trouble. |
| "Jake Paul Hoping To Gain Respect Of Boxing Community With Fight Against World's Tallest Man" | jake paul hopes to gain respect of the boxing community with fight against world's tallest man | Jake Paul hopes to gain respect of the Boxing Community with a fight against the world's tallest man. |
| "Nation Likes When Bib Has Picture Of Food They Eating" | the nation loves when a bib has a picture of food they eat | People love when Bib has a picture of food they eat. |
| "Mail Carrier Hurt To Learn Residents On His Route Have Been Receiving Electronic Mail" | Mail carrier hurt to learn residents on his route have been receiving electronic mail | Mail carrier was hurt to learn that residents on his Route have been receiving electronic mail. |

**Failure modes (RL):**
- Empty outputs for very short/vulgar inputs (e.g., "Fucker Has Nerve To Be 22 Years Old" → empty string) — reward hacking via trivially non-sarcastic empty text
- Verbatim copy for ultra-short inputs (e.g., "Norris God" → "norris god")

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

### Finding 1: Small Models Learn Surface Paraphrasing, Not Deep De-sarcasm

All three models consistently produce outputs that are **grammatically improved** versions of the input rather than semantically rewritten non-sarcastic equivalents:

```
Input:  "Area Man Passionate Defender Of What He Imagines Constitution To Be"
T5:     "Area Man Is a Passionate Defender of What He Imagines the Constitution to Be."
Target: "Local man strongly defends his personal interpretation of the Constitution"
```

The models learn low-hanging patterns: capitalize words, add articles ("a", "the"), insert punctuation. They do **not** learn to resolve the sarcastic implication.

### Finding 2: High BLEU Masks Shallow Rewriting

Because sarcastic and non-sarcastic headlines share most words, surface paraphrasing achieves deceptively reasonable BLEU scores. High word overlap ≠ successful de-sarcasm.

### Finding 3: Data Augmentation Doesn't Bridge the Gap

Adding reversed non-to-sar pairs (`--augment_reversed`) with lower word overlap (25% vs 48%) did not improve de-sarcasm quality — the models still converge on surface rewriting patterns.

### Finding 4: RL with Classifier Reward Dramatically Improves Style Transfer

Unlike SFT alone, RL with classifier reward directly optimizes for the target style. The classifier provides a training signal that cross-entropy on reference tokens cannot — it tells the model *whether the output reads as non-sarcastic*, not just whether it matches specific reference words.

**Out-of-sample evaluation on 757 sarcastic Onion headlines** (not in training data):

| Model | De-sarcasm Rate | Avg Sarcasm Prob | Identical to Input |
|-------|----------------|------------------|--------------------|
| BART-base (no fine-tuning) | 0.1% | 0.9926 | 98.7% |
| BART-base SFT (context-enhanced) | 23.9% | 0.7608 | 10.2% |
| BART-base SFT (original) | 45.3% | 0.5446 | 14.5% |
| BART-base SFT (CE) + RL | 78.9% | 0.2116 | 0.8% |
| **BART-base SFT (original) + RL** | **91.3%** | **0.0879** | **2.2%** |

*De-sarcasm Rate = fraction of outputs classified as non-sarcastic by the reward model*

RL doubles the de-sarcasm rate from SFT alone (45.3% → 91.3%), while reducing copy behavior from 14.5% to 2.2%. The average sarcasm probability of outputs drops from 0.54 to 0.09, indicating the model is not marginally passing the classifier threshold but producing outputs with high confidence of non-sarcasm.

### Finding 4b: Context-Enhanced Training Data Hurts Style Transfer

Training on context-enhanced targets (where the LLM had article bodies to produce deeper rewrites) performs *worse* than original surface-level targets, both with and without RL:

- **SFT**: CE (23.9%) < original (45.3%) — the journalistic-style targets teach the model to generate Onion-style news headlines rather than plain non-sarcastic text, which the classifier recognizes as sarcastic
- **SFT + RL**: CE (78.9%) < original (91.3%) — RL improves both, but the CE model hallucinates content-disconnected headlines (e.g., "mental hospital fire leaves hundreds of demons homeless" → "A Mental Hospital Fire Causes Widespread Damage in Georgia") while the original model stays faithful to the input

**Takeaway**: For classifier-guided RL, shallow SFT targets that keep the model close to the input provide a better foundation than deep rewrites that encourage unconstrained generation.

### Finding 5: Sarcasm is Knowledge-Intensive

| Model | Params | De-sarcasm Quality |
|-------|--------|--------------------|
| GPT-2 (SFT) | 124M | Surface paraphrasing |
| BART-base (SFT) | 139M | Surface paraphrasing |
| T5-base (SFT) | 220M | Surface paraphrasing |
| BART-base (SFT + RL) | 139M | 91.3% classifier-fooling; surface rewrites but high style accuracy |
| LLaMA 3.2 (zero-shot) | 8B | Meaningful rewrites |

De-sarcasm requires world knowledge and pragmatic reasoning. RL narrows the gap by providing a direct style signal, but fundamental comprehension still benefits from model scale.

### The Knowledge Gap

Sarcasm comprehension requires:
1. **World knowledge** — understanding what's normal vs. absurd
2. **Pragmatic inference** — recognizing speaker intent vs. literal meaning
3. **Cultural context** — knowing that TheOnion headlines follow specific comedic patterns

SFT alone can't teach these from 13K examples. RL with a classifier reward provides an orthogonal training signal — optimizing *what the output should feel like* rather than just matching reference tokens.

---

## 5. Error Taxonomy

| Error Type | Example | Frequency |
|------------|---------|-----------|
| **Capitalization-only** | "area man" → "Area Man" | High |
| **Article insertion** | "man says" → "A man says" | High |
| **Punctuation addition** | no period → added period | Medium |
| **Minor word substitution** | "passionate" → "devoted" | Low |
| **Actual de-sarcasm** | Meaningful rewrite | Rare |

---

## 6. Conclusion

- We construct a **89,688-record strategy-annotated parallel corpus** for sarcasm style transfer — a reusable resource for future work
- SFT alone on small models (124M–250M) produces **surface paraphrasing**, not genuine de-sarcasm
- **RL with classifier reward** (REINFORCE + KL penalty) provides an orthogonal training signal that pushes models toward actual style transfer — connecting our classification model (Macro F1: 0.938) directly to the generation task
- Sarcasm style transfer is **knowledge-intensive**: world knowledge and pragmatic reasoning remain bottlenecks for small models, but RL narrows the gap without requiring model scale
- **Limitation — domain specificity**: Sarcasm detection is strongly domain-dependent. Cross-domain evaluation shows neither classifier generalises well:

  | Model | NHDSD (news headlines) | iSarcasmEval (tweets) |
  |-------|----------------------|----------------------|
  | Ours (DistilBERT, trained on NHDSD) | **0.9730** | 0.4682 |
  | `cardiffnlp/twitter-roberta-base-irony` (trained on tweets) | 0.4975 | **0.6562** |

  Each model excels only in its training domain. Our classifier's 48% false positive rate on non-sarcastic tweets confirms it learned news-headline-specific patterns rather than general sarcasm. The RL reward signal is therefore calibrated to the news domain — a valid setup for Onion headlines, but not transferable to other domains without retraining the reward model
- **Future work**: LoRA fine-tuning of larger models (LLaMA 3.2 8B); DPO as an alternative RL objective; human evaluation of style transfer quality; domain-general sarcasm classifier for broader reward signal

---

## References

- Misra, R. (2019). News Headlines Dataset for Sarcasm Detection. Kaggle.
- Abu Farha, I. et al. (2022). SemEval-2022 Task 6: iSarcasmEval.
- Raffel, C. et al. (2020). T5: Exploring the Limits of Transfer Learning.
- Lewis, M. et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training.
- Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners (GPT-2).
- Zhu, K. et al. (2025). ViSP: Visual Sarcasm Generation with PPO Reinforcement Learning.
- Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.
