import Link from "next/link";

const REPO = "https://github.com/SeeYangZhi/Project-LLMao/blob/main";

const STRATEGIES = [
  {
    key: "sarcasm",
    label: "Sarcasm",
    def: "Contradicts the state of affairs with a critical tone toward the addressee.",
    example: "\"Great job breaking the build right before the demo.\"",
  },
  {
    key: "irony",
    label: "Irony",
    def: "Contradicts the state of affairs without obvious blame or target.",
    example: "\"What a beautiful day for a three-hour traffic jam.\"",
  },
  {
    key: "satire",
    label: "Satire",
    def: "Appears supportive but contains mockery that reveals absurdity.",
    example: "\"Senate Passes Landmark Bill To Study The Feasibility Of Passing Bills.\"",
  },
  {
    key: "overstatement",
    label: "Overstatement",
    def: "Obviously exaggerated terms or impossible quantities.",
    example: "\"I've told you a million times to stop exaggerating.\"",
  },
  {
    key: "understatement",
    label: "Understatement",
    def: "Severe minimization of the importance or severity of something.",
    example: "\"The Titanic experienced some minor hull damage.\"",
  },
  {
    key: "rhetorical_question",
    label: "Rhetorical Question",
    def: "A question whose expected answer contradicts reality.",
    example: "\"Is the sky blue? Obviously congress isn't corrupt.\"",
  },
];

type FileLink = { label: string; path: string };

type Stage = {
  num: string;
  title: string;
  input: string;
  inputCount: string;
  inputPath: string | null;
  process: string;
  output: string;
  outputCount: string;
  outputLinks: FileLink[];
  script: string;
  note: string;
};

const STAGES: Stage[] = [
  {
    num: "01",
    title: "Raw Collection",
    input: "NHDSD (Misra 2019)",
    inputCount: "28,619",
    inputPath: "data/raw/Sarcasm_Headlines_Dataset_v2.json",
    process: "Clean duplicates, normalize whitespace",
    output: "nhdsd_cleaned.json",
    outputCount: "28,497",
    outputLinks: [
      { label: "nhdsd_cleaned.json", path: "data/processed/intermediate/nhdsd_cleaned.json" },
    ],
    script: "reclassify_nhdsd_binary.py",
    note: "Source: TheOnion (13,634 sarcastic) + HuffPost (14,985 neutral) headlines with article links.",
  },
  {
    num: "02",
    title: "Binary Reclassification",
    input: "nhdsd_cleaned.json",
    inputCount: "28,497",
    inputPath: "data/processed/intermediate/nhdsd_cleaned.json",
    process: "LLM re-labels every headline as sarcastic / non-sarcastic",
    output: "nhdsd_reclassified.jsonl",
    outputCount: "28,497",
    outputLinks: [
      { label: "nhdsd_reclassified.jsonl", path: "data/processed/intermediate/nhdsd_reclassified.jsonl" },
      { label: "label_disagreements.jsonl", path: "data/processed/intermediate/label_disagreements.jsonl" },
    ],
    script: "StepFun 3.5 Flash · temp=0.1",
    note: "Agreement with original NHDSD labels: 80.19%. 5,644 headlines were flagged as disagreements for verification.",
  },
  {
    num: "03",
    title: "Cross-Validation",
    input: "label_disagreements.jsonl",
    inputCount: "5,644",
    inputPath: "data/processed/intermediate/label_disagreements.jsonl",
    process: "A second independent LLM re-classifies only the disagreements",
    output: "cross_validation_comparison.json",
    outputCount: "5,644",
    outputLinks: [
      { label: "cross_validation_comparison.json", path: "data/processed/intermediate/cross_validation_comparison.json" },
      { label: "cross_validation_secondary.jsonl", path: "data/processed/intermediate/cross_validation_secondary.jsonl" },
    ],
    script: "Nemotron 3 Nano 30B · temp=0.1",
    note: "Of the 5,644 disagreements, 4,076 (72.2%) had StepFun + Nemotron agreeing against the original annotation. This step produces an audit report, not a corrected labels file — the main training pipeline still reads from the raw NHDSD dataset. The CV results are consumed only by the secondary sar→non filtering pipeline.",
  },
  {
    num: "04",
    title: "Pair Generation",
    input: "Sarcasm_Headlines_Dataset_v2.json",
    inputCount: "28,619",
    inputPath: "data/raw/Sarcasm_Headlines_Dataset_v2.json",
    process: "For each headline, generate its opposite-style counterpart and tag the strategy",
    output: "sarcasm_pairs_step35_clean.jsonl",
    outputCount: "28,536",
    outputLinks: [
      { label: "sarcasm_pairs_step35_clean.jsonl", path: "data/processed/intermediate/sarcasm_pairs_step35_clean.jsonl" },
      { label: "sarcasm_pairs_non_to_sarcastic.jsonl", path: "data/processed/sarcasm_pairs_non_to_sarcastic.jsonl" },
      { label: "sarcasm_pairs_sarcastic_to_non.jsonl", path: "data/processed/sarcasm_pairs_sarcastic_to_non.jsonl" },
    ],
    script: "StepFun 3.5 Flash · temp=0.7 · batch=30",
    note: "Non-sarcastic headlines (14,948) become non→sar pairs. Sarcastic headlines (13,634) become sar→non pairs. 6 headlines hit content filters and were dropped.",
  },
  {
    num: "05",
    title: "Strategy Augmentation",
    input: "sarcasm_pairs_non_to_sarcastic.jsonl",
    inputCount: "14,948",
    inputPath: "data/processed/sarcasm_pairs_non_to_sarcastic.jsonl",
    process: "For each source, generate 5 more variants covering the missing strategies (6 total per source)",
    output: "sarcasm_pairs_non_to_sarcastic_complete.jsonl",
    outputCount: "89,688",
    outputLinks: [
      { label: "sarcasm_pairs_non_to_sarcastic_complete.jsonl", path: "data/processed/sarcasm_pairs_non_to_sarcastic_complete.jsonl" },
      { label: "sarcasm_pairs_strategy_augmented.jsonl", path: "data/processed/sarcasm_pairs_strategy_augmented.jsonl" },
    ],
    script: "StepFun 3.5 Flash · temp=0.8",
    note: "The higher temperature encourages variation across strategy types. Every source ends up with exactly 6 labeled variants — one for each sarcasm strategy.",
  },
  {
    num: "06",
    title: "Stratified Splits",
    input: "sarcasm_pairs_non_to_sarcastic_complete.jsonl",
    inputCount: "89,688",
    inputPath: "data/processed/sarcasm_pairs_non_to_sarcastic_complete.jsonl",
    process: "Source-level stratified split (seed=42) so all 6 variants of a source stay in the same split",
    output: "train / val / test",
    outputCount: "71,730 / 8,952 / 9,006",
    outputLinks: [
      { label: "train.jsonl", path: "data/splits/train.jsonl" },
      { label: "val.jsonl", path: "data/splits/val.jsonl" },
      { label: "test.jsonl", path: "data/splits/test.jsonl" },
      { label: "split_metadata.json", path: "data/splits/split_metadata.json" },
    ],
    script: "create_train_val_test_splits.py",
    note: "Source-level grouping prevents data leakage: a model can't memorize one variant of a headline and score well on another variant of the same source.",
  },
];

type SecondaryPipeline = {
  label: string;
  desc: string;
  total: string;
  splits: string;
  links: FileLink[];
};

const SECONDARY_PIPELINE: SecondaryPipeline[] = [
  {
    label: "Sarcastic → Non-sarcastic",
    desc: "After filtering to cross-validated sarcastic headlines and scraping articles to drop meme-only entries.",
    total: "13,588",
    splits: "10,868 / 1,356 / 1,364",
    links: [
      { label: "sarcasm_pairs_sar_to_non_cv_filtered.jsonl", path: "data/processed/sarcasm_pairs_sar_to_non_cv_filtered.jsonl" },
      { label: "train.jsonl", path: "data/splits/sar_to_non/train.jsonl" },
      { label: "val.jsonl", path: "data/splits/sar_to_non/val.jsonl" },
      { label: "test.jsonl", path: "data/splits/sar_to_non/test.jsonl" },
    ],
  },
  {
    label: "Context-Enhanced",
    desc: "Non-sarcastic targets generated with full article body as context. Different model (Qwen 3.6 Plus), extended strategy set (13 labels).",
    total: "10,330",
    splits: "8,258 / 1,029 / 1,043",
    links: [
      { label: "sarcasm_pairs_sar_to_non_context_enhanced.jsonl", path: "data/processed/sarcasm_pairs_sar_to_non_context_enhanced.jsonl" },
      { label: "train.jsonl", path: "data/splits/sar_to_non_context_enhanced/train.jsonl" },
      { label: "val.jsonl", path: "data/splits/sar_to_non_context_enhanced/val.jsonl" },
      { label: "test.jsonl", path: "data/splits/sar_to_non_context_enhanced/test.jsonl" },
    ],
  },
];

export default function PipelinePage() {
  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <section className="px-4 md:px-12 pt-8 md:pt-12 pb-6 md:pb-8">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3 md:mb-4"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Methodology / Data
        </span>
        <h1
          className="text-[36px] md:text-[48px] leading-[1.0] tracking-[-0.72px] md:tracking-[-0.96px] text-foreground mb-4"
          style={{ fontFamily: "var(--font-dm-serif)" }}
        >
          Data Pipeline
        </h1>
        <p className="text-[16px] md:text-[18px] leading-[1.5] text-foreground-secondary max-w-3xl">
          From 28,619 raw news headlines to 89,688 strategy-annotated training
          pairs — six stages of LLM generation, cross-validation, and stratified
          splitting designed to prevent leakage while capturing all six sarcasm
          strategies.
        </p>
      </section>

      {/* Stat ribbon */}
      <section className="px-4 md:px-12 pb-8 md:pb-12">
        <div className="border border-border-card rounded-[22px] p-5 md:p-8 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-5 md:gap-6">
          {[
            { value: "28,619", label: "Raw headlines", sub: "NHDSD v2", href: null },
            { value: "28,536", label: "Generated pairs", sub: "LLM opposites", href: null },
            { value: "89,688", label: "Augmented records", sub: "6 strategies / source", href: null },
            { value: "4,076", label: "Suspected mislabels", sub: "browse examples →", href: "/mislabels" },
            { value: "80.19%", label: "NHDSD agreement", sub: "StepFun vs original", href: null },
          ].map((stat) => {
            const inner = (
              <>
                <div
                  className="text-[32px] leading-[1.0] tracking-[-0.64px] text-foreground mb-1.5 tabular-nums group-hover:text-accent-blue transition-colors"
                  style={{ fontFamily: "var(--font-dm-serif)" }}
                >
                  {stat.value}
                </div>
                <div className="text-[14px] text-foreground-secondary">
                  {stat.label}
                </div>
                <div className="text-[12px] text-muted mt-0.5">{stat.sub}</div>
              </>
            );
            return stat.href ? (
              <Link key={stat.label} href={stat.href} className="block group">
                {inner}
              </Link>
            ) : (
              <div key={stat.label}>
                <div
                  className="text-[32px] leading-[1.0] tracking-[-0.64px] text-foreground mb-1.5 tabular-nums"
                  style={{ fontFamily: "var(--font-dm-serif)" }}
                >
                  {stat.value}
                </div>
                <div className="text-[14px] text-foreground-secondary">
                  {stat.label}
                </div>
                <div className="text-[12px] text-muted mt-0.5">{stat.sub}</div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Pipeline stages */}
      <section className="px-4 md:px-12 pb-8 md:pb-12">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-5 md:mb-6"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Primary Pipeline — Non-sarcastic → Sarcastic
        </span>
        <div className="max-w-5xl space-y-0">
          {STAGES.map((stage, i) => (
            <div key={stage.num} className="relative">
              {/* Connector line */}
              {i < STAGES.length - 1 && (
                <div className="absolute left-[28px] top-[56px] bottom-[-16px] w-px bg-border-light" />
              )}
              <div className="flex gap-6 pb-6">
                {/* Numbered dot */}
                <div className="shrink-0 w-14 h-14 rounded-full border border-border-light bg-white flex items-center justify-center">
                  <span
                    className="text-[13px] tracking-[0.16px] text-foreground-secondary"
                    style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                  >
                    {stage.num}
                  </span>
                </div>
                {/* Content */}
                <div className="flex-1 min-w-0 border border-border-card rounded-[22px] p-5 md:p-6">
                  <div className="flex flex-col md:flex-row md:items-baseline md:justify-between gap-1 mb-1">
                    <h3 className="text-[20px] md:text-[22px] tracking-[-0.22px] text-foreground">
                      {stage.title}
                    </h3>
                    <span
                      className="text-[11px] tracking-[0.16px] text-muted break-all md:break-normal"
                      style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                    >
                      {stage.script}
                    </span>
                  </div>
                  <p className="text-[14px] text-muted mb-5">
                    {stage.process}
                  </p>

                  <div className="flex flex-col md:flex-row md:items-center gap-2 md:gap-3 text-[12px] md:text-[13px] font-mono tabular-nums mb-4">
                    {stage.inputPath ? (
                      <a
                        href={`${REPO}/${stage.inputPath}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-3 py-1.5 rounded-lg bg-surface-snow text-foreground-secondary hover:text-accent-blue transition-colors"
                      >
                        <span className="text-muted">in </span>
                        {stage.inputCount}
                        <span className="text-muted"> · {stage.input}</span>
                      </a>
                    ) : (
                      <div className="px-3 py-1.5 rounded-lg bg-surface-snow text-foreground-secondary">
                        <span className="text-muted">in </span>
                        {stage.inputCount}
                        <span className="text-muted"> · {stage.input}</span>
                      </div>
                    )}
                    <span className="hidden md:inline text-muted">→</span>
                    <div className="px-3 py-1.5 rounded-lg bg-accent-blue/[0.06] text-accent-blue break-words">
                      <span className="opacity-60">out </span>
                      {stage.outputCount}
                      <span className="opacity-60"> · {stage.output}</span>
                    </div>
                  </div>

                  <p className="text-[13px] leading-[1.55] text-muted border-t border-border-card pt-4 mb-3">
                    {stage.note}
                  </p>

                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span
                      className="text-[10px] tracking-[0.16px] uppercase text-muted mr-1"
                      style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                    >
                      Files ↗
                    </span>
                    {stage.outputLinks.map((link) => (
                      <a
                        key={link.path}
                        href={`${REPO}/${link.path}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[11px] text-muted hover:text-accent-blue transition-colors px-2 py-0.5 rounded-full border border-border-card hover:border-accent-blue/30"
                        style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                      >
                        {link.label}
                      </a>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Sarcasm strategies */}
      <section className="px-4 md:px-12 pb-8 md:pb-12">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-5 md:mb-6"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Six Sarcasm Strategies
        </span>
        <p className="text-[14px] md:text-[15px] text-foreground-secondary max-w-3xl mb-5 md:mb-6">
          Adapted from the iSarcasm taxonomy. Every source headline is expanded
          into six strategy-labeled variants so models see the full distribution
          during training.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 max-w-6xl">
          {STRATEGIES.map((s) => (
            <div
              key={s.key}
              className="border border-border-card rounded-[22px] p-5 md:p-6"
            >
              <span
                className="text-[10px] tracking-[0.16px] uppercase text-muted block mb-2"
                style={{ fontFamily: "var(--font-jetbrains-mono)" }}
              >
                {s.key}
              </span>
              <h4 className="text-[18px] tracking-[-0.18px] text-foreground mb-2">
                {s.label}
              </h4>
              <p className="text-[13px] leading-[1.55] text-muted mb-4">
                {s.def}
              </p>
              <p className="text-[13px] italic text-foreground-secondary leading-[1.5] border-t border-border-card pt-3">
                {s.example}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Secondary pipelines */}
      <section className="px-4 md:px-12 pb-8 md:pb-12">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-5 md:mb-6"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Secondary Datasets
        </span>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6 max-w-5xl">
          {SECONDARY_PIPELINE.map((p) => (
            <div
              key={p.label}
              className="border border-border-card rounded-[22px] p-5 md:p-6"
            >
              <h4 className="text-[18px] tracking-[-0.18px] text-foreground mb-2">
                {p.label}
              </h4>
              <p className="text-[13px] leading-[1.55] text-muted mb-4">
                {p.desc}
              </p>
              <div className="grid grid-cols-2 gap-3 border-t border-border-card pt-4 mb-4">
                <div>
                  <div
                    className="text-[10px] tracking-[0.16px] uppercase text-muted mb-1"
                    style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                  >
                    Total
                  </div>
                  <div className="text-[16px] tabular-nums text-foreground-secondary">
                    {p.total}
                  </div>
                </div>
                <div>
                  <div
                    className="text-[10px] tracking-[0.16px] uppercase text-muted mb-1"
                    style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                  >
                    Train / Val / Test
                  </div>
                  <div className="text-[16px] tabular-nums text-foreground-secondary">
                    {p.splits}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-1.5 flex-wrap">
                <span
                  className="text-[10px] tracking-[0.16px] uppercase text-muted mr-1"
                  style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                >
                  Files ↗
                </span>
                {p.links.map((link) => (
                  <a
                    key={link.path}
                    href={`${REPO}/${link.path}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[11px] text-muted hover:text-accent-blue transition-colors px-2 py-0.5 rounded-full border border-border-card hover:border-accent-blue/30"
                    style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                  >
                    {link.label}
                  </a>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Quality controls */}
      <section className="px-4 md:px-12 pb-8 md:pb-12">
        <div className="max-w-5xl border-t border-border-light pt-6 md:pt-8">
          <span
            className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-5 md:mb-6"
            style={{ fontFamily: "var(--font-jetbrains-mono)" }}
          >
            Quality Controls
          </span>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
            <div>
              <h4 className="text-[18px] tracking-[-0.18px] text-foreground mb-2">
                Cross-validation as a data audit
              </h4>
              <p className="text-[14px] leading-[1.6] text-muted">
                Every headline StepFun disagreed with was re-checked by
                Nemotron 3 Nano 30B. When both LLMs agreed against the NHDSD
                label (4,076 of 5,644 disagreements), we recorded it as a
                suspected mislabel. These corrections are an audit signal —
                only the secondary sar→non pipeline actually filters training
                data using them. The main non→sar pipeline reads the raw
                NHDSD labels directly.
              </p>
            </div>
            <div>
              <h4 className="text-[18px] tracking-[-0.18px] text-foreground mb-2">
                Source-level stratification
              </h4>
              <p className="text-[14px] leading-[1.6] text-muted">
                Splits are done at the source headline level, not the variant
                level. All 6 strategy variants of a single source stay in the
                same split, preventing a model from memorizing one variant and
                scoring well on another.
              </p>
            </div>
            <div>
              <h4 className="text-[18px] tracking-[-0.18px] text-foreground mb-2">
                Deterministic labeling, creative generation
              </h4>
              <p className="text-[14px] leading-[1.6] text-muted">
                Classification passes run at temperature 0.1 for stability,
                while pair and variant generation run at 0.7–0.8 so the model
                produces genuinely different rewrites per strategy.
              </p>
            </div>
            <div>
              <h4 className="text-[18px] tracking-[-0.18px] text-foreground mb-2">
                Context-enhanced variant
              </h4>
              <p className="text-[14px] leading-[1.6] text-muted">
                A second smaller dataset (10,330 pairs) is generated with
                access to the full article body, producing more factually
                grounded non-sarcastic rewrites. Used to train the CE and
                CE+RL BART variants.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="px-4 md:px-12">
        <div className="max-w-5xl flex flex-col md:flex-row md:items-center md:justify-between gap-4 py-6 border-t border-border-light">
          <span className="text-[14px] text-muted">
            See how models trained on this data perform.
          </span>
          <div className="flex gap-3">
            <Link
              href="/dashboard"
              className="text-[14px] text-foreground-secondary hover:text-accent-blue transition-colors px-4 py-2 rounded-xl border border-border-card"
            >
              Model dashboard →
            </Link>
            <Link
              href="/playground"
              className="text-[14px] text-white bg-foreground hover:bg-foreground-secondary transition-colors px-4 py-2 rounded-xl"
            >
              Try the playground
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
