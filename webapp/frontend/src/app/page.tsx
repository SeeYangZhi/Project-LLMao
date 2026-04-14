import Link from "next/link";

const CARDS = [
  {
    href: "/pipeline",
    num: "01",
    title: "Data Pipeline",
    desc: "How 28,619 NHDSD headlines became 89,688 strategy-annotated training pairs through LLM generation and cross-validation.",
  },
  {
    href: "/training",
    num: "02",
    title: "Model Training",
    desc: "Exact hyperparameters and loss formulations across four recipes — SFT seq2seq, REINFORCE + KL, LoRA instruction tuning, and the 6-way ablation.",
  },
  {
    href: "/eval",
    num: "03",
    title: "Evaluation",
    desc: "What each of the 7 metrics measures, why we use it, where it breaks down. Read this before the dashboard.",
  },
  {
    href: "/dashboard",
    num: "04",
    title: "Dashboard",
    desc: "Compare 14 models across 7 evaluation metrics with interactive charts and strategy breakdowns.",
  },
  {
    href: "/explorer",
    num: "05",
    title: "Sample Explorer",
    desc: "Browse 2,857 test samples with filtering, search, and side-by-side model comparison.",
  },
  {
    href: "/playground",
    num: "06",
    title: "Playground",
    desc: "Type a sarcastic headline and watch our models rewrite it in real-time via LMStudio.",
  },
  {
    href: "/human-eval",
    num: "07",
    title: "Human Evaluation",
    desc: "140 samples × 3 models × 2 annotators (κ > 0.8). Three sarcasm classifiers all disagree with humans (κ = −0.11 to +0.18) — receipts inside.",
  },
];

export default function HomePage() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="px-4 md:px-12 pt-10 md:pt-20 pb-12 md:pb-16">
        <div className="max-w-4xl">
          <span
            className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-4 md:mb-6"
            style={{ fontFamily: "var(--font-jetbrains-mono)" }}
          >
            CS4248 / NUS / Team 14
          </span>
          <h1
            className="text-[44px] md:text-[72px] leading-[1.0] tracking-[-0.88px] md:tracking-[-1.44px] text-foreground mb-4 md:mb-6"
            style={{ fontFamily: "var(--font-dm-serif)" }}
          >
            Project LLMao
          </h1>
          <p className="text-[16px] md:text-[18px] leading-[1.5] text-foreground-secondary max-w-2xl">
            Sarcasm style transfer using fine-tuned language models. We train
            T5, BART, and LLaMA variants to rewrite sarcastic news headlines
            as neutral, factual equivalents while preserving meaning. Our
            best model (T5-Joint) achieves <span className="tabular-nums">43.6%</span>{" "}
            strict success on human evaluation.
          </p>
        </div>
      </section>

      {/* Purple band */}
      <section
        className="w-full py-12 md:py-16 px-4 md:px-12"
        style={{
          background:
            "linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 40%, #1a0a2e 100%)",
        }}
      >
        <div className="max-w-6xl grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8">
          {[
            { value: "14", label: "Models Tested", sub: "BART, T5, LLaMA" },
            { value: "2,857", label: "Test Samples", sub: "Per model" },
            { value: "3×7", label: "Classifier×Metrics", sub: "Twitter / Kaggle / News" },
            { value: "140", label: "Hand-Labeled", sub: "2 annotators, κ > 0.8" },
          ].map((stat) => (
            <div key={stat.label}>
              <div
                className="text-[36px] md:text-[48px] leading-[1.0] tracking-[-0.72px] md:tracking-[-0.96px] text-white mb-2"
                style={{ fontFamily: "var(--font-dm-serif)" }}
              >
                {stat.value}
              </div>
              <div className="text-[14px] md:text-[16px] text-white/90">
                {stat.label}
              </div>
              <div className="text-[12px] md:text-[13px] text-white/50 mt-1">
                {stat.sub}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Navigation cards */}
      <section className="px-4 md:px-12 py-12 md:py-16">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-6 md:mb-8"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Explore
        </span>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6 max-w-6xl">
          {CARDS.map((card) => (
            <Link
              key={card.href}
              href={card.href}
              className="group block border border-border-card rounded-[22px] p-6 md:p-8 transition-all duration-300 hover:border-accent-blue/30"
            >
              <span
                className="text-[11px] tracking-[0.16px] text-muted block mb-3"
                style={{ fontFamily: "var(--font-jetbrains-mono)" }}
              >
                {card.num}
              </span>
              <h3 className="text-[24px] leading-[1.3] tracking-[-0.24px] text-foreground mb-2 group-hover:text-accent-blue transition-colors">
                {card.title}
              </h3>
              <p className="text-[15px] leading-[1.5] text-muted">
                {card.desc}
              </p>
            </Link>
          ))}
        </div>
      </section>

      {/* Method overview */}
      <section className="px-4 md:px-12 pb-16 md:pb-20">
        <div className="max-w-6xl border-t border-border-light pt-10 md:pt-12">
          <span
            className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-6"
            style={{ fontFamily: "var(--font-jetbrains-mono)" }}
          >
            Methodology
          </span>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
            {[
              {
                title: "Data Pipeline",
                body: "89,688 training pairs generated from NHDSD headlines using 6 sarcasm strategies, augmented via LLM pairing with cross-validation.",
                href: "/pipeline",
              },
              {
                title: "Model Training",
                body: "Four recipes across 14 models: T5 joint-task SFT (our best, via Camille's pipeline), BART with context enhancement and REINFORCE + KL, LLaMA 3.2 1B LoRA, and a 6-way subtype ablation on T5.",
                href: "/training",
              },
              {
                title: "Evaluation",
                body: "7 automatic metrics across 3 sarcasm classifiers (RoBERTa-Twitter, Bert-Kaggle, RoBERTa-News) plus 140 hand-labeled samples. The classifiers disagree by up to 33 pp — human eval is essential.",
                href: "/human-eval",
              },
            ].map((item) => (
              <div key={item.title}>
                {item.href ? (
                  <Link
                    href={item.href}
                    className="block group"
                  >
                    <h4 className="text-[20px] tracking-[-0.2px] text-foreground mb-3 group-hover:text-accent-blue transition-colors">
                      {item.title} →
                    </h4>
                    <p className="text-[15px] leading-[1.6] text-muted">
                      {item.body}
                    </p>
                  </Link>
                ) : (
                  <>
                <h4 className="text-[20px] tracking-[-0.2px] text-foreground mb-3">
                  {item.title}
                </h4>
                <p className="text-[15px] leading-[1.6] text-muted">
                  {item.body}
                </p>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
