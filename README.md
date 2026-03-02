# Project LLMao

## Motivation

- Most existing work treats sarcasm detection as a black-box prediction task
  — the model outputs a label but offers no insight into what makes a headline sarcastic. Our project tackles the harder and more illuminating question: can we identify the specific linguistic mechanisms (hyperbole, incongruity, false sincerity, absurd specificity) that produce the sarcastic effect, and can we then use that understanding to controllably generate sarcastic/non-sarcastic text?
- A sarcastic text like "Great job on the update, everything is broken now" gets tagged as positive sentiment, leading to flawed downstream decisions. As organizations increasingly rely on automated text understanding at scale, the ability to not just detect sarcasm but understand why something reads as sarcastic becomes a practical necessity.
- This bridges interpretability and generation in a way that deepens understanding of figurative language rather than just pattern-matching surface features.
- We believe it is meaningful to be able to turn sarcastic headlines to what it truly means (the non-sarcastic version), and understand which part made the original sarcastic.

## Task Statement

- Given a sarcastic headline, generate a non-sarcastic version (or vice versa).

## Proposed Method

### Data preparation

- Pair or group headlines by topic (using similarity metrics or keyword overlap) to create pseudo-parallel sarcastic/non-sarcastic pairs.
- Based on existing NHDSD dataset, generate non-sarcastic/sarcastic versions of it.
  - Use a pretrained LLM with few-shot prompting to do style transfer between sarcastic and non-sarcastic tones.
- Classify sarcasm strategies
  — Use manual annotation to tag sarcastic headlines with their strategy type (hyperbole, incongruity, false sincerity, etc.) Prepend strategy tokens
  — Train the model with a control code: "<hyperbole> Serious headline here" → "Sarcastic version here".
- Based on the data generated
  - Fine-tune a small pretrained model on these pairs. Good candidates include T5-small or T5-base (seq2seq, natural for input→output tasks), GPT-2 (causal LM, frame it as "Serious: [input] → Sarcastic: [output]"), or BART (designed for text generation and denoising tasks). These are all small enough to fine-tune on a single GPU with modest compute.

## Proposed Evaluation

- Use both automatic metrics (BLEU, perplexity) and a simple human evaluation (or classifier-based evaluation — feed generated headlines into your sarcasm detector to check if they're actually detected as sarcastic).

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
