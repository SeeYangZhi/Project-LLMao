# Evaluation

> Metrics, evaluation protocols, and pipeline for Project LLMao.

## Evaluation Pipeline

The main evaluation pipeline (`scripts/eval_pipeline.py`) computes 7 automated metrics plus an LLM judge, with human evaluation support.

### Quick Start

```bash
# Dry run with synthetic data
python scripts/dry_run_eval.py
python scripts/eval_pipeline.py --input dry_run_input.csv --output dry_run_results.csv --skip_judge
# With real iSarcasmEval data
python scripts/convert_isarcasm.py
python scripts/eval_pipeline.py --input isarcasm_test.csv --output isarcasm_results.csv --skip_judge
# With LLM judge (requires Gemini API key)
python scripts/eval_pipeline.py --input data.csv --output results.csv --gemini_key AIza...
# Full options
python scripts/eval_pipeline.py \
    --input data.csv \
    --output results.csv \
    --gemini_key AIza... \
    --judge_sample 50 \
    --human_eval_n 30 \
    --references gold_references.csv
```

### Input Format

CSV with columns: `[id, input, output, subtype]`

- `id`: Sample identifier
- `input`: Original sarcastic text
- `output`: Model-generated non-sarcastic rewrite
- `subtype`: Sarcasm strategy (sarcasm, irony, satire, understatement, overstatement, rhetorical_question)

## Automatic Metrics

### Metric 1: Sarcasm Flip Rate

Uses `cardiffnlp/twitter-roberta-base-irony` classifier.

- **Hard Flip Rate**: % of samples where input classified sarcastic AND output classified non-sarcastic
- **Mean Flip Delta**: mean(input_irony_score - output_irony_score); positive = tone shifted toward non-sarcastic
  **Target**: Higher flip rate and positive flip delta = better sarcasm removal.

### Metric 2: Semantic Similarity

Uses `all-MiniLM-L6-v2` sentence-transformers with cosine similarity.

- Measures whether the rewrite preserves the original meaning
- **Target**: > 0.6 (same topic/meaning preserved)

### Metric 3: Fluency (Perplexity)

Uses GPT-2 language model perplexity.

- **Target**: Lower is better (more fluent)
- Typical range: 10-100 for news-style text

### Metric 4: BLEU Score

Uses NLTK sentence BLEU with smoothing.

- **vs input mode** (default): Measures how much the output differs from the input. Lower = more rewriting done (desirable).
- **vs reference mode**: Standard BLEU against gold rewrites. Higher = closer to gold.

### Metric 5: LLM-as-Judge (Gemini)

Batch evaluation using Gemini 2.5 Flash. Rates each pair on:
| Dimension | Scale | Description |
| ------------------ | ----- | ------------------------------------ |
| sarcasm_removed | 1-5 | Was sarcasm successfully removed? |
| meaning_preserved | 1-5 | Does output keep same topic/meaning? |
| fluency | 1-5 | Is output natural English? |
**Cohen's Kappa**: Compares judge binary (sarcasm_removed >= 4 → 1) vs classifier hard_flipped. Two independent raters on same samples. > 0.6 = trustworthy agreement.

### Metric 6: Edit Distance (Word-Level)

Word-level Levenshtein distance between input and output tokens.

- **Raw**: Absolute number of word insertions/deletions/substitutions
- **Normalized**: Raw / max(len(input), len(output)), range 0-1
- **Target**: Higher normalized distance = more substantial rewriting
- Complements BLEU (order-sensitive) with order-independent word change measurement

### Metric 7: Paraphrase Score

Combined score: `semantic_similarity × BLEU_vs_input`

- **High score** = output has similar meaning AND similar words → paraphrasing (bad)
- **Low score** = genuine rewriting with different words → actual sarcasm removal (good)
- Directly quantifies the paraphrasing failure mode observed in fine-tuned models

## Human Evaluation

### Automated Flagging

The pipeline auto-exports the top N most "suspicious" samples for manual review:

- **Suspicion score** = paraphrase_score + (1 - |flip_delta|)
- Flags: `high_paraphrase`, `low_flip_delta`, `very_similar_wording`
- Output: `{results}_human_eval.csv` with empty annotator columns

### Protocol

1. 2-3 team members independently label each flagged sample:
   - `sarcasm_removed`: yes/no
   - `meaning_preserved`: yes/no
2. Compute inter-annotator Cohen's kappa between human raters
3. Compare human labels against LLM judge to validate automated evaluation
   This addresses the instructor feedback: "otherwise you have no incentive to study the individual outputs yourselves as humans."

## Metrics Summary

| Metric              | What It Measures         | Target | Script        |
| ------------------- | ------------------------ | ------ | ------------- |
| Flip Rate           | Sarcasm removal          | Higher | eval_pipeline |
| Semantic Similarity | Meaning preservation     | > 0.6  | eval_pipeline |
| Perplexity          | Fluency                  | Lower  | eval_pipeline |
| BLEU (vs input)     | Rewriting degree         | Lower  | eval_pipeline |
| BLEU (vs reference) | Gold similarity          | Higher | eval_pipeline |
| Edit Distance       | Word-level changes       | Higher | eval_pipeline |
| Paraphrase Score    | Paraphrasing detection   | Lower  | eval_pipeline |
| LLM Judge           | Multi-dimensional rating | Higher | eval_pipeline |
| METEOR              | Flexible overlap         | Higher | eval          |

## iSarcasmEval Baseline Numbers (Gold Human Rephrases)

These serve as reference points — model outputs should be compared against these:
| Metric | Gold Baseline |
| ---------------------- | ------------- |
| Hard Flip Rate | 46.14% |
| Mean Flip Delta | +0.0089 |
| Semantic Similarity | 0.5625 |
| BLEU (vs input) | 0.1018 |
| Perplexity (GPT-2) | 230.85 |
| Edit Distance (norm) | 0.8187 |
| Paraphrase Score | 0.0757 |
**Key insight**: If fine-tuned models show paraphrase score >> 0.0757 or edit distance << 0.82, they are paraphrasing more than humans do.

## Reproducibility

- Use fixed random seed for sampling
- Report number of samples evaluated
- Include sample outputs in appendix
- Log Gemini model version used for LLM judge

---

_Last updated: 2026-04-05_
