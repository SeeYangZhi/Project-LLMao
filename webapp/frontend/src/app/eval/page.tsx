import Link from "next/link";

type ScoreBand = { range: string; meaning: string; tone?: "good" | "ok" | "bad" };

type ToolLink = { label: string; href: string };

type MetricDoc = {
  num: string;
  key: string;
  name: string;
  tool: string;
  question: string;
  why: string;
  bands?: ScoreBand[];
  limitation?: string;
  note?: string;
  toolLinks?: ToolLink[];
};

const METRICS: MetricDoc[] = [
  {
    num: "01",
    key: "flip-rate",
    name: "Sarcasm Flip Rate",
    tool: "RoBERTa-Twitter · Bert-Kaggle · RoBERTa-News",
    toolLinks: [
      {
        label: "cardiffnlp/twitter-roberta-base-irony",
        href: "https://huggingface.co/cardiffnlp/twitter-roberta-base-irony",
      },
      {
        label: "helinivan/english-sarcasm-detector",
        href: "https://huggingface.co/helinivan/english-sarcasm-detector",
      },
      {
        label: "jkhan447/sarcasm-detection-RoBerta-base-POS",
        href: "https://huggingface.co/jkhan447/sarcasm-detection-RoBerta-base-POS",
      },
    ],
    question: "Does the classifier think sarcasm was removed?",
    why: "The most direct measure of task success — was the output reclassified as non-sarcastic? We run all three classifiers because they trained on different domains (Twitter irony tweets, a Kaggle headlines dataset, news headlines) and disagree by up to 33 percentage points on the same outputs.",
    bands: [
      { range: "Higher", meaning: "More outputs flagged non-sarcastic", tone: "good" },
    ],
    limitation:
      "All three classifiers fail against human ground truth (Cohen's κ from −0.11 to +0.18 vs human κ > 0.8). They detect sarcasm presence in isolation, not removal between input and output. Treat as one signal, not as ground truth.",
  },
  {
    num: "02",
    key: "similarity",
    name: "Semantic Similarity",
    tool: "sentence-transformers/all-MiniLM-L6-v2",
    question: "Is the core meaning preserved between input and output?",
    why: "Sarcasm style transfer must keep the underlying claim. Cosine similarity on sentence embeddings is the standard cheap proxy for semantic preservation across rewrites.",
    bands: [
      { range: "0.95+", meaning: "Nearly identical (possibly just paraphrased)", tone: "ok" },
      { range: "0.85–0.95", meaning: "Good meaning preservation", tone: "good" },
      { range: "0.70–0.85", meaning: "Moderate drift", tone: "ok" },
      { range: "< 0.70", meaning: "Significant meaning loss", tone: "bad" },
    ],
    limitation:
      "Doesn't detect copying — a model that just lowercases the input scores 0.99. Always pair with BLEU vs input.",
  },
  {
    num: "03",
    key: "perplexity",
    name: "Perplexity",
    tool: "GPT-2",
    question: "Is the output fluent, natural English?",
    why: "Catches degenerate outputs (truncations, gibberish, broken syntax) that other metrics might miss. We use GPT-2 because it's a fixed reference LM that pre-dates our training data.",
    bands: [
      { range: "< 300", meaning: "Very fluent", tone: "good" },
      { range: "300–600", meaning: "Normal", tone: "good" },
      { range: "600–1000", meaning: "Somewhat disfluent", tone: "ok" },
      { range: "> 1000", meaning: "Problematic", tone: "bad" },
    ],
    limitation:
      "Mean perplexity is dominated by long-tail outliers (a single broken sample drags the average up). The dashboard reports the mean — flag outliers rather than treat the absolute value as authoritative.",
  },
  {
    num: "04",
    key: "bleu",
    name: "BLEU vs Input",
    tool: "sacrebleu",
    question: "How much n-gram overlap is there between output and input?",
    why: "Detects whether the model is genuinely rewriting or just copying the input back. We compare against the input (not a reference) because we want to penalize models that take shortcuts.",
    bands: [
      { range: "High BLEU + High sim", meaning: "Paraphrasing — minimal real change", tone: "ok" },
      { range: "Low BLEU + High sim", meaning: "Genuine rewriting", tone: "good" },
      { range: "Low BLEU + Low sim", meaning: "Meaning lost", tone: "bad" },
    ],
    limitation:
      "Only meaningful in combination with similarity. Low BLEU alone could mean either successful rewriting or content destruction.",
  },
  {
    num: "05",
    key: "edit-dist",
    name: "Edit Distance",
    tool: "Word-level Levenshtein, normalized to [0, 1]",
    question: "How much was the text modified?",
    why: "Complementary to BLEU — measures structural change, not just n-gram overlap. Useful for separating models that delete tokens (high edit distance, low BLEU) from models that paraphrase (moderate edit distance, moderate BLEU).",
    bands: [
      { range: "0.0–0.3", meaning: "Minor edits (punctuation, casing)", tone: "ok" },
      { range: "0.4–0.6", meaning: "Moderate rewriting", tone: "good" },
      { range: "0.7–0.9", meaning: "Significant rewriting", tone: "good" },
      { range: "0.9+", meaning: "Complete rewrite", tone: "ok" },
    ],
    limitation:
      "Doesn't tell you whether the rewriting was good — just how much there was. Use alongside similarity.",
  },
  {
    num: "06",
    key: "llm-judge",
    name: "LLM-as-Judge",
    tool: "Gemini 2.5 Flash",
    question: "What does a strong LLM think of the output across three dimensions?",
    why: "Captures what surface metrics miss. We score each output 1–5 on (a) sarcasm_removed — is the output non-sarcastic? (b) meaning_preserved — is the core claim intact? (c) fluency — is it natural English? Run on a 50-sample batch per model to keep cost bounded.",
    bands: [
      { range: "5.0", meaning: "Strong agreement with human intent", tone: "good" },
      { range: "3.0–4.0", meaning: "Mixed signal", tone: "ok" },
      { range: "< 3.0", meaning: "Failed dimension", tone: "bad" },
    ],
    limitation:
      "Expensive to run at full scale and known to be biased (LLMs prefer LLM-style outputs). Used as a sample evaluation, not a primary metric.",
  },
  {
    num: "07",
    key: "paraphrase",
    name: "Paraphrase Score",
    tool: "similarity × (1 − BLEU vs input)",
    question: "Is the model rewriting genuinely while still preserving meaning?",
    why: "Existing metrics fail individually — high similarity alone doesn't catch copying, low BLEU alone doesn't distinguish rewriting from destruction. Paraphrase score multiplies them so a model has to score well on BOTH to win.",
    bands: [
      { range: "> 0.70", meaning: "Strong rewriting — preserves meaning and diverges from input", tone: "good" },
      { range: "0.60–0.70", meaning: "Moderate — where most fine-tuned models land", tone: "good" },
      { range: "0.50–0.60", meaning: "Around the human-baseline level", tone: "ok" },
      { range: "< 0.50", meaning: "Below human baseline — copying or meaning loss", tone: "bad" },
    ],
    note:
      "Concrete example. Input: \"Man Shocked By Obvious Fact\". Output A: \"man shocked by obvious fact\" → similarity 0.99, BLEU 0.95 → paraphrase 0.05 (just copied). Output B: \"A person was surprised to learn something widely known\" → similarity 0.85, BLEU 0.08 → paraphrase 0.78 (genuine rewrite). The gold human rewrites from iSarcasmEval land around 0.51 on this metric.",
  },
];

export default function EvalPage() {
  return (
    <div className="min-h-screen pb-16 md:pb-20">
      {/* Header */}
      <section className="px-4 md:px-12 pt-8 md:pt-12 pb-6 md:pb-8">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3 md:mb-4"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Methodology
        </span>
        <h1
          className="text-[36px] md:text-[48px] leading-[1.0] tracking-[-0.72px] md:tracking-[-0.96px] text-foreground mb-4"
          style={{ fontFamily: "var(--font-dm-serif)" }}
        >
          Evaluation
        </h1>
        <p className="text-[16px] md:text-[18px] leading-[1.55] text-foreground-secondary max-w-3xl">
          Why we measure each metric, what its score means, and where it
          breaks down. Every model in the dashboard reports these seven
          numbers — this page is the rosetta stone that explains them.
        </p>
      </section>

      {/* Reading guide */}
      <section className="px-4 md:px-12 pb-8 md:pb-10">
        <div className="border border-border-card rounded-[22px] p-5 md:p-6 max-w-3xl">
          <h3 className="text-[16px] md:text-[18px] tracking-[-0.18px] text-foreground mb-3">
            How to read these together
          </h3>
          <ul className="text-[13px] md:text-[14px] leading-[1.7] text-foreground-secondary space-y-2 list-disc pl-5">
            <li>
              <strong className="text-foreground">No metric is sufficient on its own.</strong> A
              model that scores best on flip rate often scores worst on
              meaning. Read the row, not the column.
            </li>
            <li>
              <strong className="text-foreground">Pair similarity with BLEU vs input.</strong>{" "}
              High similarity + high BLEU = copying. High similarity + low BLEU
              = genuine rewriting. The paraphrase score formalises this.
            </li>
            <li>
              <strong className="text-foreground">Treat flip rate as one signal, not ground truth.</strong>{" "}
              The classifiers disagree with each other by 33 pp and with humans
              by κ ≈ 0.10. Cross-check with the human eval.
            </li>
          </ul>
        </div>
      </section>

      {/* Metric cards */}
      <section className="px-4 md:px-12 pb-10 md:pb-12">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-4 md:mb-6"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          The 7-Metric Pipeline
        </span>
        <div className="space-y-4 md:space-y-5 max-w-4xl">
          {METRICS.map((m) => (
            <article
              key={m.key}
              id={m.key}
              className="border border-border-card rounded-[22px] p-5 md:p-7"
            >
              <header className="mb-4 md:mb-5">
                <div className="flex items-baseline gap-3 mb-2">
                  <span
                    className="text-[11px] tracking-[0.16px] text-muted"
                    style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                  >
                    {m.num}
                  </span>
                  <h3 className="text-[20px] md:text-[24px] tracking-[-0.24px] text-foreground">
                    {m.name}
                  </h3>
                </div>
                <p
                  className="text-[12px] md:text-[13px] text-muted"
                  style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                >
                  {m.tool}
                </p>
                {m.toolLinks && (
                  <ul className="mt-2 space-y-1">
                    {m.toolLinks.map((link) => (
                      <li
                        key={link.href}
                        className="text-[11px] md:text-[12px]"
                        style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                      >
                        <a
                          href={link.href}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-accent-blue hover:underline"
                        >
                          {link.label} ↗
                        </a>
                      </li>
                    ))}
                  </ul>
                )}
              </header>

              <div className="grid grid-cols-1 md:grid-cols-[140px_1fr] gap-2 md:gap-4 mb-4 md:mb-5">
                <div className="text-[11px] tracking-[0.28px] uppercase text-muted">
                  Question
                </div>
                <div className="text-[14px] md:text-[15px] leading-[1.6] text-foreground">
                  {m.question}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-[140px_1fr] gap-2 md:gap-4 mb-4 md:mb-5">
                <div className="text-[11px] tracking-[0.28px] uppercase text-muted">
                  Why we use it
                </div>
                <div className="text-[14px] md:text-[15px] leading-[1.6] text-foreground-secondary">
                  {m.why}
                </div>
              </div>

              {m.bands && (
                <div className="grid grid-cols-1 md:grid-cols-[140px_1fr] gap-2 md:gap-4 mb-4 md:mb-5">
                  <div className="text-[11px] tracking-[0.28px] uppercase text-muted">
                    Score interpretation
                  </div>
                  <ul className="space-y-1.5">
                    {m.bands.map((b) => (
                      <li
                        key={b.range}
                        className="flex items-baseline gap-3 text-[13px] md:text-[14px]"
                      >
                        <span
                          className={`tabular-nums shrink-0 w-32 ${
                            b.tone === "good"
                              ? "text-green-600"
                              : b.tone === "bad"
                              ? "text-red-500"
                              : b.tone === "ok"
                              ? "text-amber-600"
                              : "text-foreground-secondary"
                          }`}
                          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                        >
                          {b.range}
                        </span>
                        <span className="text-foreground-secondary">{b.meaning}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {m.limitation && (
                <div className="grid grid-cols-1 md:grid-cols-[140px_1fr] gap-2 md:gap-4">
                  <div className="text-[11px] tracking-[0.28px] uppercase text-amber-600">
                    Limitation
                  </div>
                  <div className="text-[13px] md:text-[14px] leading-[1.6] text-foreground-secondary">
                    {m.limitation}
                  </div>
                </div>
              )}

              {m.note && (
                <div className="mt-4 md:mt-5 pt-4 border-t border-border-card text-[12px] md:text-[13px] leading-[1.6] text-muted">
                  {m.note}
                </div>
              )}
            </article>
          ))}
        </div>
      </section>

      {/* Human eval section */}
      <section
        id="human-eval"
        className="px-4 md:px-12 pb-12 md:pb-16"
      >
        <div className="border-t border-border-light pt-10 md:pt-12 max-w-4xl">
          <span
            className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3 md:mb-4"
            style={{ fontFamily: "var(--font-jetbrains-mono)" }}
          >
            Section 02
          </span>
          <h2
            className="text-[28px] md:text-[36px] leading-[1.05] tracking-[-0.36px] md:tracking-[-0.54px] text-foreground mb-4 md:mb-5"
            style={{ fontFamily: "var(--font-dm-serif)" }}
          >
            Human Evaluation
          </h2>
          <p className="text-[15px] md:text-[17px] leading-[1.6] text-foreground-secondary mb-6 md:mb-8 max-w-3xl">
            Why we did it: every automated metric has known failure modes, and
            our multi-classifier audit showed the flip rate disagrees with
            itself by up to 33 percentage points. We needed a ground truth.
            Two annotators independently labeled 140 stratified samples per
            model on two binary questions: <em>sarcasm removed?</em> and{" "}
            <em>meaning changed?</em>
          </p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-5 mb-8 md:mb-10">
            {[
              { v: "140", l: "Samples per model", s: "Stratified by subtype" },
              { v: "3", l: "Models annotated", s: "T5-Joint, T5-Control, BART-RL" },
              { v: "2", l: "Independent annotators", s: "Per model" },
              { v: "κ > 0.8", l: "Inter-annotator", s: "Excellent agreement" },
            ].map((stat) => (
              <div key={stat.l}>
                <div
                  className="text-[28px] md:text-[36px] leading-[1.0] tracking-[-0.56px] md:tracking-[-0.72px] text-foreground mb-1.5 tabular-nums"
                  style={{ fontFamily: "var(--font-dm-serif)" }}
                >
                  {stat.v}
                </div>
                <div className="text-[13px] md:text-[14px] text-foreground-secondary">
                  {stat.l}
                </div>
                <div className="text-[11px] md:text-[12px] text-muted mt-0.5">
                  {stat.s}
                </div>
              </div>
            ))}
          </div>

          <h3 className="text-[18px] md:text-[20px] tracking-[-0.2px] text-foreground mb-3">
            What we found
          </h3>
          <ul className="text-[14px] md:text-[15px] leading-[1.7] text-foreground-secondary space-y-2 list-disc pl-5 mb-6 md:mb-8">
            <li>
              <strong className="text-foreground">All three classifiers fail.</strong>{" "}
              Cohen&apos;s κ vs human ranges from <span className="tabular-nums">−0.11</span>{" "}
              (Bert-Kaggle on T5-Control) to{" "}
              <span className="tabular-nums">+0.18</span> (RoBERTa-News on
              T5-Joint). Four of the nine model×classifier cells show negative
              κ — the classifier anti-correlates with humans.
            </li>
            <li>
              <strong className="text-foreground">T5-Joint is the best model overall.</strong>{" "}
              Strict success rate (sarcasm removed AND meaning preserved) is{" "}
              <span className="tabular-nums">43.6%</span> — beating T5-Control{" "}
              (<span className="tabular-nums">39.3%</span>) and BART-RL (
              <span className="tabular-nums">34.3%</span>). The strategy prefix
              forces task decomposition before generation.
            </li>
            <li>
              <strong className="text-foreground">BART-RL destroys meaning.</strong>{" "}
              Meaning-change rate is <span className="tabular-nums">40.7%</span>{" "}
              — more than double T5-Joint&apos;s{" "}
              <span className="tabular-nums">16.4%</span>. The reward function
              rewards deletion of sarcastic tokens, not faithful rewriting.
            </li>
            <li>
              <strong className="text-foreground">Subtypes fail differently.</strong>{" "}
              Satire has the highest classifier miss rate (80%) because it
              mimics legitimate news format. Rhetorical questions encode
              sarcasm in implication, not lexical markers. Overstatement is a{" "}
              <em>model</em> failure — the model can&apos;t remove what defines
              the headline.
            </li>
          </ul>

          <div className="border border-accent-blue/30 bg-accent-blue/[0.03] rounded-[22px] p-5 md:p-6">
            <h4 className="text-[15px] md:text-[16px] text-foreground mb-2">
              The full evidence
            </h4>
            <p className="text-[13px] md:text-[14px] leading-[1.6] text-foreground-secondary mb-4">
              The Human Eval page has the receipts: per-model summary cards,
              the 9-cell classifier-vs-human accuracy table, per-subtype miss
              rates, and the multi-classifier comparison across all 14 models.
            </p>
            <Link
              href="/human-eval"
              className="text-[14px] text-accent-blue hover:underline"
            >
              Open Human Eval →
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
