# Dataset Documentation

> Data sources, schemas, and preprocessing for Project LLMao.

## Primary Dataset: NHDSD

**Name**: News Headlines Dataset for Sarcasm Detection
**Source**: Kaggle - https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
**Size**: ~28,000 news headlines
**Type**: Text Classification / Generation

### Schema

```json
{
  "is_sarcastic": 1, // 1 = sarcastic, 0 = non-sarcastic
  "headline": "string", // News headline text
  "article_link": "url" // Original article URL
}
```

### Data Sources

| Source    | Type          | Count      |
| --------- | ------------- | ---------- |
| TheOnion  | Sarcastic     | 13,634     |
| HuffPost  | Non-sarcastic | 14,985     |
| **Total** |               | **28,619** |

### Characteristics

- **Professional writing**: No spelling mistakes, formal language
- **Self-contained**: No reply context required
- **High-quality labels**: TheOnion = intentional sarcasm
- **Topic-matched**: Headlines cover similar events

## Reference Datasets

### iSarcasmEval

**GitHub**: https://github.com/iabufarha/iSarcasmEval
**Purpose**: Additional sarcasm data with strategy annotations

### Sarcasm Corpus V2

**GitHub**: https://github.com/soraby/sarcasm2
**Purpose**: Reference dataset for comparison

## Preprocessing Pipeline

### Step 1: Data Cleaning

```python
# Remove duplicates
# Normalize whitespace
# Handle special characters
```

### Step 2: LLM Generation

Generate opposite-style headlines using MiniMax M2.5:

- For each sarcastic headline → generate non-sarcastic version
- For each non-sarcastic headline → generate sarcastic version
- Total: 28,619 pairs (bidirectional)

### Step 3: Strategy Annotation

Tag each sarcastic headline with strategy (from iSarcasm dataset):

| Strategy                | Description                                | Example                                              |
| ----------------------- | ------------------------------------------ | ---------------------------------------------------- |
| `<sarcasm>`             | Contradicts state of affairs, critical     | "Great job on the update, everything is broken now"  |
| `<irony>`               | Contradicts state of affairs, not critical | "I love waiting in line at the DMV"                  |
| `<satire>`              | Appears to support, contains mockery       | "Wow, another meeting that could have been an email" |
| `<understatement>`      | Undermines importance                      | "It's just a minor setback"                          |
| `<overstatement>`       | Obviously exaggerated                      | "I've told you a million times!"                     |
| `<rhetorical_question>` | Question contradicting reality             | "Isn't it just the best feeling to be ignored?"      |

### Step 4: Augmentation (Optional)

Use LLM with few-shot prompting to generate additional pairs:

- Prompt: `Sarcastic: {input}\nNon-sarcastic: {output}`
- Few-shot with 3-5 examples

### Step 5: Train/Val/Test Split

```
train: 80%
val:   10%
test:  10%
```

Stratify by:

- Sarcasm label (sarcastic/non-sarcastic)
- Strategy type (if annotated)

## Data Directory Structure

```
data/
├── raw/
│   ├── nhdsd/
│   │   └── Sarcasm_Headlines_Dataset_v2.json
│   ├── isarcasm/
│   │   └── ...
│   └── sarcasm2/
│       └── ...
├── processed/
│   ├── paired/
│   │   └── paired_headlines.json
│   ├── annotated/
│   │   └── strategy_annotated.json
│   └── augmented/
│       └── llm_augmented.json
└── splits/
    ├── train.json
    ├── val.json
    └── test.json
```

## Data Versioning

Track dataset versions:

- Use SHA or date in filenames
- Log version in experiment configs
- Document any filtering/cleaning steps

---

_Last updated: 2026-03-02_
