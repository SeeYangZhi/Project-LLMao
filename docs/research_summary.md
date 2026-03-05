# Research Summary

> Pre-implementation literature and methodology review for CS4248 Sarcasm Classification Project.

---

## 1. Why TF-IDF + Logistic Regression Is a Strong Baseline

### TF-IDF as Feature Representation

TF-IDF (Term Frequency–Inverse Document Frequency) transforms raw text into a sparse vector space where each dimension corresponds to a vocabulary token. The weight of a term is:

$$\text{tfidf}(t,d) = \text{tf}(t,d) \times \log\frac{N}{1 + \text{df}(t)}$$

- **Term frequency** rewards tokens that appear often in a document
- **Inverse document frequency** down-weights tokens common across all documents, promoting discriminative terms

For sarcasm detection, TF-IDF captures **lexical sarcasm markers**: exaggerated praise terms, specific ironic vocabulary ("absolutely brilliant," "clearly"), and domain-specific token patterns from TheOnion-style writing.

### Logistic Regression on Sparse TF-IDF

Logistic Regression is particularly well-suited to high-dimensional sparse features:
- **Linear separator in feature space**: Sufficient for many text classification tasks where class-specific vocabulary is the primary signal
- **Regularization (L2 default)**: Prevents overfitting on rare n-grams
- **Calibrated probabilities**: Useful for error analysis and threshold tuning
- **Interpretable weights**: Top positive/negative coefficients reveal discriminative lexical features
- **Scalable**: Handles 28k+ samples and 50k+ feature dimensions efficiently

**Reference practice**: Joachims (1998) established SVM/linear models on TF-IDF as a strong text classification baseline; Logistic Regression performs comparably with faster convergence.

### N-gram Variants

- **Unigrams (1,1)**: Capture individual sarcasm-associated words
- **Bigrams (1,2)**: Capture collocations ("not exactly," "of course") that signal irony
- **Character n-grams (3,5)**: Capture morphological and stylistic signals (unusual capitalization artifacts, punctuation sequences)

---

## 2. Why Naive Bayes Is a Valid Baseline for Sparse Text

### Generative Model with Multinomial Assumption

MultinomialNB models the probability of a class given token counts:
$$P(y | \mathbf{x}) \propto P(y) \prod_i P(x_i | y)^{x_i}$$

This is optimal when features are conditionally independent—an assumption that doesn't hold perfectly in text but approximates well for bag-of-words features.

### Advantages for This Task

- **Sample efficient**: Performs well with limited training data due to the generative model structure
- **Computationally trivial**: Training is a single pass over data; inference is extremely fast
- **Robust to sparse features**: No gradient instability issues with high-dimensional sparse inputs
- **Smoothing parameter (alpha)**: Laplace/Lidstone smoothing prevents zero-probability tokens

### CountVectorizer vs TF-IDF for Naive Bayes

- **CountVectorizer + MultinomialNB**: The multinomial likelihood is derived assuming raw count features; theoretically correct
- **TF-IDF + MultinomialNB**: Not theoretically motivated (TF-IDF can produce non-integer "counts") but often performs well empirically by reducing the influence of high-frequency stop words
- **Comparison is necessary**: Empirical results in the literature are mixed; we run both

**Reference**: McCallum & Nigam (1998) showed Naive Bayes competitive with SVM on text classification; Rennie et al. (2003) proposed Complement NB for class imbalance.

---

## 3. Why BERT/DistilBERT Is Suitable for Sarcasm Detection

### Contextual Representations

Unlike TF-IDF which produces context-free token weights, BERT produces **context-sensitive embeddings** via multi-head self-attention. This is critical for sarcasm where:
- The same word can be sarcastic or literal depending on surrounding context
- Irony often relies on incongruence that requires understanding the full sentence
- Subtle pragmatic cues ("what a surprise") require modeling over the full token sequence

### Pre-training on Large Corpora

BERT is pre-trained on Books Corpus + Wikipedia (3.3B words), giving it:
- General syntactic and semantic knowledge
- Common knowledge about the world (useful for incongruence detection)
- Sub-word tokenization (WordPiece) robust to rare vocabulary

### Sarcasm-specific Relevance

- **Rhetorical questions** ("Who wouldn't want that?"): Self-attention captures the interrogative-declarative incongruence
- **Overstatement/understatement**: Sentiment-aware representations from pre-training encode hyperbole cues
- **Irony/satire**: Semantic incongruence between entities and predicates is detectable via attention patterns

### DistilBERT vs BERT-base

| Model | Parameters | Inference Speed | Accuracy Retention |
|-------|-----------|-----------------|-------------------|
| DistilBERT-base-uncased | 66M | ~60% faster than BERT | ~97% of BERT |
| BERT-base-uncased | 110M | Baseline | Baseline |

**Strategy**: Start with DistilBERT for rapid iteration; upgrade to BERT-base if compute budget allows.

**References**: Devlin et al. (2018) BERT; Sanh et al. (2019) DistilBERT; Joshi et al. (2022) for sarcasm classification with transformers.

---

## 4. Why Macro-F1 Matters

### Binary Task

The binary dataset (after expansion) has classes from two `type` values:
- `sarcastic_to_non`: 13,408 pairs → contributes 13,408 sarcastic + 13,408 non-sarcastic
- `non_to_sarcastic`: 14,925 pairs → contributes 14,925 sarcastic + 14,925 non-sarcastic

Binary classes are **approximately balanced** (both classes ~28k total), so weighted-F1 ≈ macro-F1 here. However, we still report macro-F1 as primary for consistency.

### Type Task (Multiclass — Critical)

The 6 strategy classes are **significantly imbalanced**:
| Strategy | Count (approx) |
|----------|---------------|
| sarcasm | ~8,699 |
| irony | ~6,102 |
| satire | ~5,224 |
| overstatement | ~3,976 |
| understatement | ~3,295 |
| rhetorical_question | ~1,037 |

A model that always predicts "sarcasm" achieves ~31% accuracy but 0% macro-F1 on minority classes.

**Macro-F1 = unweighted average of per-class F1** ensures minority classes like `rhetorical_question` have equal weight in the optimization objective.

---

## 5. Leakage Risks in Paired Transformation Datasets

### The Risk

This dataset was constructed by:
1. Taking real sarcastic headlines (TheOnion) → generating non-sarcastic rewrites
2. Taking real non-sarcastic headlines (HuffPost) → generating sarcastic rewrites

Each JSONL row contains a **semantically similar pair**: the sarcastic and non-sarcastic versions share topic, entities, and much of their lexical content.

**If we split after expansion**: A train set containing `"Scientists unveil doomsday clock"` (sarcastic) will directly leak topic/entity information about `"Scientists present research on hair loss"` (non-sarcastic) which may be in the test set.

The model will learn spurious correlations between specific entity patterns and labels rather than the true pragmatic sarcasm signal.

### Mitigation Strategy

**Pair-level splitting** (our primary approach):
1. Assign `pair_id = row_index` to each JSONL row
2. Assign `group_id` = normalized `article_link` (captures rows with same source article)
3. Split at the **group level before expansion**
4. All rows from a group appear in exactly one of {train, val, test}

This guarantees:
- Both the sarcastic and non-sarcastic versions of a pair go to the same split
- No topic/entity overlap between train and test

**Expected conservative result**: Group-level splitting may slightly reduce apparent performance compared to naive splitting, reflecting true generalization.

---

## 6. Implementation Strategy

### Execution Order (Fast → Slow)

1. **Dataset audit** (15 min): Validate schema, derive labels, check distributions
2. **Split generation** (5 min): Group-safe train/val/test splits, save to disk
3. **TF-IDF + LR** (30 min): Grid search over n-grams, min_df, C, class_weight
4. **Naive Bayes** (20 min): CountVectorizer vs TF-IDF variants
5. **DistilBERT** (hours): Fine-tuning with early stopping
6. **BERT-base** (optional, hours): Only if compute allows
7. **Error analysis + comparison** (1 hour): Report generation

### Primary Assumptions

- **No aggressive text cleaning**: Preserve punctuation (sarcasm cues), lowercase for classical models, raw text for BERT tokenizer
- **macro-F1 is primary metric** for all model selection decisions
- **Group integrity > stratification**: If exact stratified splits are impossible under grouping constraints, preserve groups and document class distributions
- **All artifacts saved to disk**: No results exist only in memory; all runs are reproducible

---

## References

1. Joachims, T. (1998). Text categorization with support vector machines. ECML.
2. McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification. AAAI.
3. Rennie, J., et al. (2003). Tackling the poor assumptions of Naive Bayes. ICML.
4. Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers. NAACL.
5. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. NeurIPS workshop.
6. Joshi, A., et al. (2022). Sarcasm detection using deep learning. Survey.
7. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR.
