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

## Project-Specific Guidelines

### Model Selection

| Model   | When to Use                     |
| ------- | ------------------------------- |
| T5-base | Primary model for seq2seq tasks |
| GPT-2   | Baseline for causal LM framing  |
| BART    | Baseline for denoising approach |

### Data Handling

- Use NHDSD as primary dataset (~28K headlines)
- Augment with LLM-generated style transfers
- Annotate with strategy labels

### Evaluation Protocol

1. **Automatic**: BLEU score against reference translations
2. **Perplexity**: Measure fluency
3. **Classifier**: Feed to sarcasm detector, check detection rate
4. **Human**: Rate sarcasm effectiveness and naturalness

### CS4248 Context

This is a university course project. Key implications:

- Focus on demonstrating learning (2W1H: Why, What, How)
- SOTA results not required - show deep understanding
- Document all AI tool use
- Ablation studies recommended to understand components

## Anti-Patterns

- **Pure classification**: Don't just detect - explain
- **Black-box generation**: Don't use uncontrolled text-to-text
- **No logging**: Always track experiments

---

_Last updated: 2026-03-02_
