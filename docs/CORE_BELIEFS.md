# Core Beliefs

> Design philosophy and golden principles for Project LLMao.

## Golden Principles

### 1. No Black-Box Outputs

**Belief**: Understanding _why_ sarcasm works is as important as detecting it.

**Practice**:

- Always identify the linguistic mechanism (sarcasm, irony, satire, etc. - see iSarcasm categories)
- Use strategy tokens to make reasoning explicit
- Interpret outputs in terms of the strategy used

**Why**: Sarcasm detection as pure classification misses the "how" and "why" - the linguistic mechanisms that produce the sarcastic effect. This project bridges interpretability and generation.

### 2. Strategy-First Approach

**Belief**: Classify the sarcasm strategy _before_ generating.

**Practice**:

- Annotate training data with strategy labels
- Use control codes: `<sarcasm>`, `<irony>`, `<satire>`, `<understatement>`, `<overstatement>`, `<rhetorical_question>`
- Generate: `<strategy> input` → `output`

**Why**: Different strategies require different linguistic transformations. Treating them uniformly loses interpretability.

### 3. Controllable Generation

**Belief**: Generation should be interpretable and controllable.

**Practice**:

- Use explicit control codes in input
- Log which strategy was used for each output
- Avoid purely learned latent variables

**Why**: Controllable outputs allow us to understand the model and debug failures.

### 4. Reproducibility

**Belief**: Every experiment must be reproducible.

**Practice**:

- Log all hyperparameters, seeds, dataset versions
- Store model checkpoints with config
- Document data preprocessing steps

**Why**: Research requires reproducibility. Future work must be able to build on this.

### 5. Human Evaluation as Ground Truth

**Belief**: When automated metrics disagree with human judgment,
trust the humans.

**Practice**:

- Hand-label a stratified sample (140 headlines × 3 models, 2 annotators)
- Compute inter-annotator κ to verify the labels are reliable (ours: > 0.8)
- Treat any classifier whose κ vs human is below ~0.3 as unreliable
- Never present a single classifier's flip rate as a primary metric

**Why**: Our multi-classifier audit shows three sarcasm classifiers
disagree by up to 33 percentage points on the same outputs and all
three score κ between −0.11 and +0.18 against human labels. Picking
any one of them as "the" flip rate would have given us a confidently
wrong story about which model is best. Human eval revealed T5-Joint
(not BART-RL, the apparent winner by classifier flip rate) is the
strongest model.

## Project-Specific Guidelines

### Model Selection

| Model family | Role |
|---|---|
| T5-Joint (T5-base) | Best on human eval; strategy-prefix joint task |
| T5-Control (T5-base) | Same recipe minus the joint prefix — isolates the contribution |
| BART-Base / BART-CE | SFT baselines, with and without article-context data |
| BART-RL / BART-CE+RL | REINFORCE + KL refinement on the SFT checkpoints |
| LLaMA 3.2 1B (LoRA) | Larger backbone; aggressive rewriter, hallucination risk |
| 6 ablation models | T5-base × 6, each with one subtype dropped |

### Data Handling

- NHDSD as primary dataset (28,619 headlines, dedup → 28,497)
- StepFun 3.5 Flash + Nemotron 3 Nano for synthetic pair generation and
  cross-validation; 4,076 NHDSD headlines flagged as suspected mislabels
  (browsable in the webapp `/mislabels` page)
- Strategy augmentation produces 89,688 records — 6 strategy variants per
  source headline, balanced across all six subtypes

### Evaluation Protocol

1. **7-metric pipeline** on the full 2,857-sample test split: flip rate
   (× 3 classifiers), similarity, perplexity, BLEU, edit distance,
   paraphrase score, LLM-as-judge
2. **Multi-classifier audit**: report all three classifier flip rates,
   not a single number. Spread is the diagnostic.
3. **Human evaluation**: 140 stratified samples × 3 models × 2 annotators
4. **Reproducibility**: every metric backed by a script in `scripts/`
   and a commit-pinned data file in `results/`

### CS4248 Context

This is a university course project. Key implications:

- Focus on demonstrating learning (2W1H: Why, What, How)
- SOTA results not required - show deep understanding
- Document all AI tool use
- Ablation studies recommended to understand components

## Anti-Patterns

- **Pure classification**: Don't just detect — explain
- **Black-box generation**: Don't use uncontrolled text-to-text
- **No logging**: Always track experiments
- **Single-classifier flip rate**: Don't report a flip rate without
  declaring the classifier and ideally citing all three for spread
- **"BLEU went up" framing**: BLEU vs input alone is meaningless —
  always pair with similarity (the paraphrase score formalises this)

---

_Last updated: 2026-04-14_
