# Report Guide

> Guide for writing the CS4248 final report based on the course template.

## Report Overview

**Page Limit**: 8 pages (main body), excluding references, statements, and appendices.

### Excluded from Page Limit
- References
- Statement of Independent Work
- Ethical Statement (optional)
- Appendices

### Format Requirements

| Element | Specification |
|---------|---------------|
| Format | Two-column, A4 |
| Font | Adobe Times Roman |
| Margins | 2.5 cm (all sides) |
| Submission | PDF via Canvas |

### Font Sizes

| Element | Font Size |
|---------|-----------|
| Paper Title | 15 pt Bold |
| Author names | 12 pt Bold |
| Section titles | 12 pt Bold |
| Document text | 11 pt |
| Captions | 11 pt |
| Abstract text | 10 pt |
| Bibliography | 10 pt |
| Footnotes | 9 pt |

## Report Structure

### Section 1: Abstract (100-200 words)

**Purpose**: Summary of entire work

Include:
- Task statement
- Experimental highlights
- Key findings
- One sentence on significance

### Section 2: Introduction (0.5-1 page)

**Purpose**: Motivate and frame the problem

Include:
- Problem motivation (why sarcasm matters)
- Clear task statement
- Key contributions (3-4 bullet points)
- Overview of approach

**Tip**: Define inputs/outputs clearly in NLP terminology

### Section 3: Related Work / Background (0.5-1.5 pages)

**Purpose**: Contextualize and identify gaps

Include:
- Prior sarcasm detection work
- Style transfer methods
- Identify what existing work misses
- Datasets used in field

### Section 4: Corpus Analysis & Method (1-2 pages)

**Purpose**: Describe data and approach

Include:
- Dataset statistics and characteristics
- Preprocessing pipeline
- Model architecture choices
- Justify why approach fits task

**Tip**: Include a diagram of your pipeline

### Section 5: Experiments (1-2 pages)

**Purpose**: Show what you did and results

Include:
- Experimental settings (hyperparameters, seeds)
- Evaluation metrics
- Baselines compared
- Results in tables/figures

**Tip**: Tables should be captioned and referenced

### Section 6: Discussion (1-3 pages)

**Purpose**: Deep analysis

Answer 2-3 research questions:
1. Does strategy-aware generation improve quality?
2. How do different models compare?
3. What are failure modes?

Include:
- Error analysis
- Performance on sub-populations
- Compute costs
- At least one question on **natural language aspect**

### Section 7: Conclusion (0.25-0.5 pages)

**Purpose**: Summarize and look forward

Include:
- Key insights
- Limitations
- Future directions

## Marking Rubric (100 Marks)

### Presentation (25%)
- Motivation: Clear goals, NLP terminology
- Structure: Logical flow
- Visualization: Effective figures/tables
- Prose: Integrated text and visuals

### Content (60%)
- Originality: Novel elements
- Technical Justification: Approach validity
- Implementation: Multiple models, clean code
- Model Evaluation: Macroscopic + microscopic
- Results Interpretation: Error analysis

### Miscellaneous (15%)
- Reproducibility: Detailed approach
- Limitations: Honest assessment
- Compliance: Proper statements filed

## Key Guidelines

### Focus on Learning, Not SOTA

> "Technical complexity is not the main focus; emphasis is on the 'learning' and the '2W1H' (Why, What, How)."

- Show deep understanding of your approach
- Better to use simpler models with clear explanation than black-box SOTA

### Ablation Studies

Recommended to understand component contributions:
- With/without strategy control codes
- Different model architectures
- Data size variations

### AI Tool Documentation

Required: Document all AI tool use
- Prompts used
- Outputs received
- How you verified/corrected outputs

**Location**: Audit trail or appendix

### What's Expected

| Aspect | Expectation |
|--------|-------------|
| Performance | Not SOTA required |
| Focus | Learning demonstration |
| Models | Baseline vs. best comparison |
| Evaluation | Automated + human metrics |
| Analysis | Error analysis required |

## Writing Tips

### Do
- Use clear, formal academic language
- Define technical terms
- Cite prior work (use natbib)
- Show concrete examples

### Don't
- Don't treat sarcasm as black-box
- Don't skip human evaluation
- Don't omit error analysis
- Don't overclaim results

## Submission Checklist

- [ ] 8-page limit (main body)
- [ ] PDF format
- [ ] Statement of Independent Work (signed)
- [ ] AI tool usage documented
- [ ] References (any length)
- [ ] Ethical Statement (if applicable)

---

*Last updated: 2026-03-02*
