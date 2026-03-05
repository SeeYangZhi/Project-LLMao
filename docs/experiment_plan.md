# Experiment Plan

> Execution plan for sarcasm classification pipeline.
> Written: 2026-03-03

---

## 1. Split Strategy

### Group-Aware Holdout Split

**Grouping key**: `group_id = normalize(article_link)` (fallback: `pair_id`)

**Split ratios** (at group level):
| Split | Groups | Approx Samples (binary) |
|-------|--------|------------------------|
| Train | 70% | ~39,666 |
| Val | 15% | ~8,500 |
| Test | 15% | ~8,500 |

**Implementation**:
1. Assign `pair_id` = row index (0-based)
2. Assign `group_id` = normalized `article_link` (strip protocol, lowercase, strip trailing slash)
3. Get unique `group_id` values; shuffle with fixed seed (SEED=42)
4. Split groups 70/15/15
5. Expand each group's JSONL rows into binary samples; all samples from a group go to the same split

**Stratification note**: Exact stratification by `type` at group level is approximate; we shuffle with a fixed seed and document the resulting class distribution. Binary task is balanced by construction, so stratification is less critical here. For type task, document per-class counts in each split.

### Cross-Validation (Classical Models)

**Method**: GroupKFold (k=5) on training+val groups
- Groups: same `group_id` as holdout split
- Use `cross_val_score` with `GroupKFold` for hyperparameter selection
- Report mean ± std macro-F1 across folds
- Refit best config on full train+val; evaluate on held-out test

---

## 2. Metrics (All Models, Both Tasks)

For each model and each task (binary + type), report:

| Metric | Notes |
|--------|-------|
| Accuracy | Overall correct / total |
| Precision (macro) | Unweighted average of per-class precision |
| Recall (macro) | Unweighted average of per-class recall |
| **F1 (macro)** | **PRIMARY model-selection metric** |
| F1 (weighted) | Class-frequency-weighted F1 |
| Confusion matrix | Saved as PNG and CSV |
| Per-class P/R/F1 | From `classification_report` |

### Model Selection Criterion

- **Binary task**: Validation macro-F1 (primary), weighted-F1 (secondary)
- **Type task**: Validation macro-F1 (primary)

---

## 3. Hyperparameter Search Spaces

### 3.1 TF-IDF + Logistic Regression

**Binary Task** (full grid, ~36 combinations):

| Parameter | Values |
|-----------|--------|
| `tfidf__ngram_range` | `(1,1)`, `(1,2)` |
| `tfidf__min_df` | `2`, `3`, `5` |
| `tfidf__max_features` | `None`, `50000` |
| `lr__C` | `0.1`, `1.0`, `3.0` |
| `lr__class_weight` | `None` |

Char n-gram variant (separate run):
| Parameter | Values |
|-----------|--------|
| `tfidf__analyzer` | `char_wb` |
| `tfidf__ngram_range` | `(3,5)` |
| `lr__C` | `0.1`, `1.0`, `3.0` |

**Type Task** (same grid, add class_weight):

| Parameter | Values |
|-----------|--------|
| `tfidf__ngram_range` | `(1,1)`, `(1,2)` |
| `tfidf__min_df` | `2`, `3`, `5` |
| `tfidf__max_features` | `None`, `50000` |
| `lr__C` | `0.1`, `1.0`, `3.0` |
| `lr__class_weight` | `None`, `balanced` |

**Search method**: `GridSearchCV` with `GroupKFold(5)`, scoring=`f1_macro`

### 3.2 Naive Bayes

**Both Tasks**:

| Parameter | CountVectorizer | TF-IDF |
|-----------|----------------|--------|
| `ngram_range` | `(1,1)`, `(1,2)` | `(1,1)`, `(1,2)` |
| `min_df` | `2`, `3`, `5` | `2`, `3`, `5` |
| `nb__alpha` | `0.1`, `0.5`, `1.0` | `0.1`, `0.5`, `1.0` |

Two classifier variants:
- `MultinomialNB` (primary)
- `ComplementNB` (nice-to-have, better for imbalanced type task)

**Search method**: `GridSearchCV` with `GroupKFold(5)`, scoring=`f1_macro`

### 3.3 DistilBERT / BERT

| Parameter | Values |
|-----------|--------|
| `model_name` | `distilbert-base-uncased` (primary), `bert-base-uncased` (optional) |
| `max_length` | `128` |
| `batch_size` | `16`, `32` |
| `learning_rate` | `2e-5`, `3e-5`, `5e-5` |
| `epochs` | Up to 10 (with early stopping) |
| `warmup_ratio` | `0.1` |
| `weight_decay` | `0.01` |
| `seed` | `42` |

**Early stopping**: patience=3 on validation macro-F1 (binary) or macro-F1 (type)

**Class weighting for type task**:
- Compute `class_weight` = `len(y_train) / (n_classes * np.bincount(y_train))`
- Apply as `weight` tensor in `CrossEntropyLoss`

**Training runs** (in order):
1. DistilBERT, binary task, default LR
2. DistilBERT, type task, with class weights
3. BERT-base, binary task (optional)
4. BERT-base, type task (optional)

---

## 4. Output Artifact Locations

### Datasets
```
outputs/datasets/
├── binary_dataset.csv          # Full expanded binary dataset
├── type_dataset.csv            # Sarcastic-only type dataset
├── splits/
│   ├── train_binary.csv
│   ├── val_binary.csv
│   ├── test_binary.csv
│   ├── train_type.csv
│   ├── val_type.csv
│   └── test_type.csv
```

### Classical Model Outputs
```
outputs/classical/
├── tfidf_lr/
│   ├── best_config_binary.json
│   ├── best_config_type.json
│   ├── metrics_binary.json
│   ├── metrics_type.json
│   ├── predictions_val_binary.csv
│   ├── predictions_test_binary.csv
│   ├── predictions_val_type.csv
│   ├── predictions_test_type.csv
│   ├── confusion_matrix_binary.png
│   └── confusion_matrix_type.png
├── naive_bayes/
│   ├── best_config_binary.json
│   ├── best_config_type.json
│   ├── metrics_binary.json
│   ├── metrics_type.json
│   ├── predictions_val_binary.csv
│   ├── predictions_test_binary.csv
│   ├── predictions_val_type.csv
│   ├── predictions_test_type.csv
│   ├── confusion_matrix_binary.png
│   └── confusion_matrix_type.png
```

### BERT Outputs
```
outputs/bert/
├── distilbert_binary/
│   ├── config.json
│   ├── best_checkpoint/         # HuggingFace model checkpoint
│   ├── training_log.csv
│   ├── metrics.json
│   ├── predictions_val.csv
│   ├── predictions_test.csv
│   └── confusion_matrix.png
├── distilbert_type/
│   └── ...
├── bert_base_binary/            (optional)
└── bert_base_type/              (optional)
```

### Reports
```
outputs/reports/
├── model_comparison.md
├── error_analysis.md
├── error_examples_binary.csv
└── error_examples_type.csv
```

---

## 5. Execution Order (Fastest → Slowest)

| Step | Notebook | Estimated Time | Prerequisite |
|------|----------|---------------|-------------|
| 1 | `01_data_preparation.ipynb` | 5-10 min | None |
| 2 | `02_tfidf_lr_baseline.ipynb` | 15-30 min | Step 1 |
| 3 | `03_naive_bayes_baseline.ipynb` | 10-20 min | Step 1 |
| 4 | `04_bert_classification.ipynb` | 2-6 hours | Steps 2-3 |
| 5 | `05_error_analysis.ipynb` | 30-60 min | Steps 2-4 |

---

## 6. Success Criteria

### Binary Task

| Model | Acceptable macro-F1 | Good macro-F1 |
|-------|---------------------|--------------|
| TF-IDF + LR | > 0.75 | > 0.85 |
| Naive Bayes | > 0.70 | > 0.80 |
| DistilBERT | > 0.85 | > 0.90 |

### Type Task (Multiclass)

| Model | Acceptable macro-F1 | Good macro-F1 |
|-------|---------------------|--------------|
| TF-IDF + LR | > 0.40 | > 0.60 |
| Naive Bayes | > 0.35 | > 0.55 |
| DistilBERT | > 0.55 | > 0.70 |

Type task is harder due to:
- 6-class imbalanced distribution
- Strategy overlap (sarcasm vs irony vs satire)
- Subtle pragmatic differences between strategies

### Final Model Selection

Best model for recommendation = highest test macro-F1 on the primary metric.

If DistilBERT test macro-F1 ≤ best classical macro-F1 by < 0.02: recommend classical model (simpler, faster).

---

## 7. Reproducibility Checklist

- [ ] `SEED = 42` used everywhere (numpy, random, sklearn, torch, transformers)
- [ ] All split files saved to disk before model training
- [ ] All model configs saved to JSON before training
- [ ] All metrics saved to JSON after evaluation
- [ ] All predictions saved to CSV
- [ ] All confusion matrices saved as PNG
- [ ] BERT checkpoints saved; best checkpoint logged
- [ ] `requirements.txt` pinned

---

## Key Assumptions Documented

1. **Grouping**: `article_link` is the primary grouping key; rows with same article_link share a group. Two observed duplicate article_links are handled correctly.
2. **Text normalization**: Lowercase for classical models; raw text (preserve case/punctuation) for BERT tokenizer.
3. **No aggressive cleaning**: Punctuation is preserved (sarcasm cues).
4. **50/50 binary balance**: The expansion of 28,333 pairs produces exactly 28,333 sarcastic + 28,333 non-sarcastic samples.
5. **`is_generated` flag**: Included in datasets to enable artifact-aware error analysis.
6. **Cross-validation uses train+val groups only**: Test groups are never seen during hyperparameter selection.
