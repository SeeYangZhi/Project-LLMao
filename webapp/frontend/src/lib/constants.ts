export const MODEL_COLORS: Record<string, string> = {
  bart_base: "#1a1a2e",
  bart_base_ce: "#16213e",
  bart_base_rl: "#0f3460",
  bart_base_ce_rl: "#1863dc",
  llama_3_2_1b: "#9b60aa",
  t5_base_joint: "#059669",
  t5_control: "#10b981",
  joint: "#6ee7b7",
  ablation_without_sarcasm: "#f59e0b",
  ablation_without_irony: "#f97316",
  ablation_without_satire: "#ef4444",
  ablation_without_overstatement: "#ec4899",
  ablation_without_understatement: "#8b5cf6",
  ablation_without_rhetorical_question: "#06b6d4",
};

export const METRIC_INFO: Record<
  string,
  { label: string; description: string; higher_better: boolean; format: string }
> = {
  hard_flip_rate: {
    label: "Flip Rate (Twitter)",
    description: "% flagged non-sarcastic by RoBERTa-Twitter — see Human Eval for the other 2 classifiers and human ground truth",
    higher_better: true,
    format: "percent",
  },
  flip_rate_kaggle: {
    label: "Flip Rate (Kaggle)",
    description: "% flagged non-sarcastic by Bert-Kaggle (helinivan/english-sarcasm-detector) — wildly different from the other classifiers",
    higher_better: true,
    format: "percent",
  },
  flip_rate_news: {
    label: "Flip Rate (News)",
    description: "% flagged non-sarcastic by RoBERTa-News — closest to human judgment but still κ ≈ 0.10",
    higher_better: true,
    format: "percent",
  },
  flip_delta: {
    label: "Flip Delta",
    description: "Average change in irony score (higher = better removal)",
    higher_better: true,
    format: "decimal",
  },
  similarity: {
    label: "Similarity",
    description: "Semantic similarity between input and output (meaning preservation)",
    higher_better: true,
    format: "decimal",
  },
  bleu: {
    label: "BLEU",
    description: "BLEU score vs input (lower = more rewriting, which is better)",
    higher_better: false,
    format: "decimal",
  },
  perplexity: {
    label: "Perplexity",
    description: "GPT-2 perplexity, mean — sensitive to long-tail outliers",
    higher_better: false,
    format: "decimal",
  },
  edit_dist_norm: {
    label: "Edit Distance",
    description: "Normalized word-level edit distance (higher = more rewriting)",
    higher_better: true,
    format: "decimal",
  },
  paraphrase_score: {
    label: "Paraphrase Score",
    description: "similarity × (1 − BLEU). Higher = genuine rewriting that still preserves meaning.",
    higher_better: true,
    format: "decimal",
  },
};

export const STRATEGIES = [
  { key: "sarcasm", label: "Sarcasm", color: "#1863dc" },
  { key: "irony", label: "Irony", color: "#9b60aa" },
  { key: "satire", label: "Satire", color: "#0f3460" },
  { key: "overstatement", label: "Overstatement", color: "#f59e0b" },
  { key: "understatement", label: "Understatement", color: "#06b6d4" },
  { key: "rhetorical_question", label: "Rhetorical Q", color: "#ef4444" },
];

export function formatMetric(value: number, format: string): string {
  if (format === "percent") return `${value.toFixed(1)}%`;
  if (value > 100) return value.toFixed(0);
  return value.toFixed(4);
}
