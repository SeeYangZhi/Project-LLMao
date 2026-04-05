# Project LLMao

## Motivation

- Most existing work treats sarcasm detection as a black-box prediction task
  — the model outputs a label but offers no insight into what makes a headline sarcastic. Our project tackles the harder and more illuminating question: can we identify the specific linguistic mechanisms (hyperbole, incongruity, false sincerity, absurd specificity) that produce the sarcastic effect, and can we then use that understanding to controllably generate sarcastic/non-sarcastic text?
- A sarcastic text like "Great job on the update, everything is broken now" gets tagged as positive sentiment, leading to flawed downstream decisions. As organizations increasingly rely on automated text understanding at scale, the ability to not just detect sarcasm but understand why something reads as sarcastic becomes a practical necessity.
- This bridges interpretability and generation in a way that deepens understanding of figurative language rather than just pattern-matching surface features.
- We believe it is meaningful to be able to turn sarcastic headlines to what it truly means (the non-sarcastic version), and understand which part made the original sarcastic.

## Task Statement

- Given a sarcastic headline, generate a non-sarcastic version that conveys the same underlying meaning.
- Secondary direction: given a non-sarcastic headline and a strategy control code, generate a sarcastic version using that specific strategy.

## Method

### Data Preparation: Synthetic Parallel Corpus Construction

No large-scale paired sarcasm style transfer dataset exists, so we use an LLM (StepFun Step-3.5 Flash via OpenRouter) to construct a synthetic parallel corpus. This follows the established paradigm of using capable models for dataset creation (Taori et al., 2023; Li et al., 2023). A secondary model (Nemotron) cross-validates label quality on disagreements.

- Starting from the NHDSD dataset (28,619 headlines), generate opposite-style counterparts using few-shot prompting
- Annotate each sarcastic variant with one of 6 strategy control codes: `<sarcasm>`, `<irony>`, `<satire>`, `<understatement>`, `<overstatement>`, `<rhetorical_question>`
- Augment each source with 5 additional strategy variants for complete coverage
- Result: 89,688 strategy-annotated paired records, split 80/10/10 at source level

### Why Large LLMs for Preprocessing, Small Models for the Task

The LLM serves strictly as a **synthetic data annotator** — it creates the training signal that doesn't exist in the wild, analogous to human annotation at scale. The research contribution is whether small, efficient models can learn **controllable, strategy-aware** style transfer from this synthetic supervision. This distinction is:

- **Scientific**: A fine-tuned T5/GPT-2/BART is inspectable and allows ablation of control codes. The LLM is a black box.
- **Practical**: A fine-tuned T5-base runs inference in ~10ms on a single GPU. An LLM API call costs per-token and takes 1-2s. For any deployable application (content moderation, sentiment correction), a small model is necessary.
- **Controllable**: Our models respond to explicit strategy control codes for deterministic, strategy-specific outputs. The LLM does not offer this structured control.

### Model Training

Primary focus: **sarcastic → non-sarcastic** (de-sarcasm), with secondary experiments on non-sarcastic → sarcastic using strategy control codes.

Fine-tune pretrained models on the synthetic parallel pairs:
- **T5-base** (220M, seq2seq): `<strategy> source_headline` → `target_headline`
- **GPT-2** (124M, causal LM): `<strategy> source_headline → target_headline`
- **BART-base** (139M, denoising seq2seq): same framing as T5, plus RL refinement (REINFORCE + KL penalty)
- **Llama-3.2-1B-Instruct** (1.24B, causal LM): LoRA fine-tuning (r=16, α=32) with instruct chat template; trained on context-enhanced data with article bodies

Small models (T5/BART/GPT-2) are fine-tuned on a single GPU. LLaMA uses LoRA (11.3M trainable params, 0.9% of total) on H200 GPU.

## Proposed Evaluation

- Use both automatic metrics (BLEU, perplexity) and a simple human evaluation (or classifier-based evaluation — feed generated headlines into your sarcasm detector to check if they're actually detected as sarcastic).

## Project Structure

```
Project LLMao/
├── notebooks/                  # Jupyter notebooks (classification pipeline)
├── scripts/                    # Training, eval, and SLURM scripts
├── scripts/data_prep/          # Completed data processing pipeline
├── data/
│   ├── raw/                    # Original NHDSD dataset (28,619 headlines)
│   ├── processed/              # Final generated datasets (89,688 records)
│   │   └── intermediate/       # Pipeline artifacts
│   └── splits/                 # Train/val/test splits (80/10/10)
├── docs/                       # Architecture, methods, evaluation, etc.
├── AGENTS.md                   # Project guide & conventions
├── README.md
└── pyproject.toml
```

# Context of Main Dataset:

- Name: News Headlines Dataset for Sarcasm Detection (NHDSD)
- Type: Text Classification (or Generation)
- Size: ~ 28,000 news headlines
- Link: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

## Details of Dataset:

- Each record consists of three attributes:
  - is_sarcastic: 1 if the record is sarcastic otherwise 0
  - headline: the headline of the news article
  - article_link: link to the original news article. Useful in collecting supplementary data

## Description of Dataset

- Past studies in Sarcasm Detection mostly make use of Twitter datasets collected using hashtag based supervision but such datasets are noisy in terms of labels and language. Furthermore, many tweets are replies to other tweets and detecting sarcasm in these requires the availability of contextual tweets.

- To overcome the limitations related to noise in Twitter datasets, this News Headlines dataset for Sarcasm Detection is collected from two news website. TheOnion aims at producing sarcastic versions of current events and we collected all the headlines from News in Brief and News in Photos categories (which are sarcastic). We collect real (and non-sarcastic) news headlines from HuffPost.

- This new dataset has following advantages over the existing Twitter datasets:

- Since news headlines are written by professionals in a formal manner, there are no spelling mistakes and informal usage. This reduces the sparsity and also increases the chance of finding pre-trained embeddings.

- Furthermore, since the sole purpose of TheOnion is to publish sarcastic news, we get high-quality labels with much less noise as compared to Twitter datasets.

- Unlike tweets which are replies to other tweets, the news headlines we obtained are self-contained. This would help us in teasing apart the real sarcastic elements.

# Context of Reference Dataset #1:

- Name: iSarcasmEval Dataset
- Link: https://github.com/iabufarha/iSarcasmEval

# Context of Reference Dataset #2:

- Name: Sarcasm Corpus V2
- Link: https://github.com/soraby/sarcasm2
