import Link from "next/link";

const CARDS = [
  {
    href: "/dashboard",
    num: "01",
    title: "Dashboard",
    desc: "Compare 14 models across 7 evaluation metrics with interactive charts and strategy breakdowns.",
  },
  {
    href: "/explorer",
    num: "02",
    title: "Sample Explorer",
    desc: "Browse 2,857 test samples with filtering, search, and side-by-side model comparison.",
  },
  {
    href: "/playground",
    num: "03",
    title: "Playground",
    desc: "Type a sarcastic headline and watch our models rewrite it in real-time via LMStudio.",
  },
  {
    href: "/human-eval",
    num: "04",
    title: "Human Evaluation",
    desc: "Gold standard annotations, flagged samples, and inter-annotator agreement analysis.",
  },
];

export default function HomePage() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="px-12 pt-20 pb-16">
        <div className="max-w-4xl">
          <span
            className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-6"
            style={{ fontFamily: "var(--font-jetbrains-mono)" }}
          >
            CS4248 / NUS / Team 14
          </span>
          <h1
            className="text-[72px] leading-[1.0] tracking-[-1.44px] text-foreground mb-6"
            style={{ fontFamily: "var(--font-dm-serif)" }}
          >
            Project LLMao
          </h1>
          <p className="text-[18px] leading-[1.5] text-foreground-secondary max-w-2xl">
            Sarcasm style transfer using fine-tuned language models. We train
            BART and LLaMA variants to rewrite sarcastic news headlines as
            neutral, factual equivalents while preserving meaning.
          </p>
        </div>
      </section>

      {/* Purple band */}
      <section
        className="w-full py-16 px-12"
        style={{
          background:
            "linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 40%, #1a0a2e 100%)",
        }}
      >
        <div className="max-w-6xl grid grid-cols-4 gap-8">
          {[
            { value: "14", label: "Models Tested", sub: "BART, T5, LLaMA" },
            { value: "2,857", label: "Test Samples", sub: "Per model" },
            { value: "7", label: "Auto Metrics", sub: "Flip rate, BLEU, ..." },
            { value: "6", label: "Strategies", sub: "Sarcasm subtypes" },
          ].map((stat) => (
            <div key={stat.label}>
              <div
                className="text-[48px] leading-[1.0] tracking-[-0.96px] text-white mb-2"
                style={{ fontFamily: "var(--font-dm-serif)" }}
              >
                {stat.value}
              </div>
              <div className="text-[16px] text-white/90">{stat.label}</div>
              <div className="text-[13px] text-white/50 mt-1">{stat.sub}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Navigation cards */}
      <section className="px-12 py-16">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-8"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Explore
        </span>
        <div className="grid grid-cols-2 gap-6 max-w-6xl">
          {CARDS.map((card) => (
            <Link
              key={card.href}
              href={card.href}
              className="group block border border-border-card rounded-[22px] p-8 transition-all duration-300 hover:border-accent-blue/30"
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
      <section className="px-12 pb-20">
        <div className="max-w-6xl border-t border-border-light pt-12">
          <span
            className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-6"
            style={{ fontFamily: "var(--font-jetbrains-mono)" }}
          >
            Methodology
          </span>
          <div className="grid grid-cols-3 gap-8">
            {[
              {
                title: "Data Pipeline",
                body: "89,688 training pairs generated from NHDSD headlines using 6 sarcasm strategies, augmented via LLM pairing with cross-validation.",
              },
              {
                title: "Model Training",
                body: "BART with context enhancement and REINFORCE + KL penalty. LLaMA 3.2 1B with LoRA fine-tuning. T5 baselines and ablation studies.",
              },
              {
                title: "Evaluation",
                body: "7 automatic metrics including sarcasm flip rate, semantic similarity, perplexity, and paraphrase detection. Gemini LLM-as-judge + human annotation.",
              },
            ].map((item) => (
              <div key={item.title}>
                <h4 className="text-[20px] tracking-[-0.2px] text-foreground mb-3">
                  {item.title}
                </h4>
                <p className="text-[15px] leading-[1.6] text-muted">
                  {item.body}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
