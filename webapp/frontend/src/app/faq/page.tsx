import Link from "next/link";

type FaqItem = {
  num: string;
  source: "instructor" | "mentor";
  question: string;
  answer: React.ReactNode;
  links?: { label: string; href: string; external?: boolean }[];
};

type FaqSection = {
  title: string;
  blurb?: string;
  items: FaqItem[];
};

const SECTIONS: FaqSection[] = [
  {
    title: "Methodology & framing",
    blurb:
      "Responses to interim feedback on how the task is set up and how we justify the two-stage pipeline.",
    items: [
      {
        num: "01",
        source: "instructor",
        question:
          "Why use larger LLMs for preprocessing but smaller models (T5, BART, LLaMA) as the actual task target? Isn't that backwards?",
        answer: (
          <>
            <p className="mb-3">
              The goal of LLMao is to ship a <em>small, cheap</em> inference-time
              model — that is literally the &quot;Lightweight Language Models&quot;
              in the project name. The two stages have very different cost
              profiles:
            </p>
            <ul className="list-disc ml-5 space-y-1.5 mb-3">
              <li>
                <strong>Preprocessing is a one-time offline cost.</strong> We
                call the teacher models once to build the training corpus
                (89,688 strategy-annotated pairs from 28,619 NHDSD headlines).
                After that, the teachers are never called again.
              </li>
              <li>
                <strong>The target model is used forever at inference.</strong>{" "}
                T5-Joint (220M) and LLaMA 3.2 1B run on commodity hardware —
                a phone, a laptop, a free-tier LMStudio instance — without an
                API key.
              </li>
            </ul>
            <p className="mb-3">
              Concretely the preprocessing pipeline uses two OpenRouter models
              on the free tier. <strong>Step 3.5 Flash</strong>{" "}
              (<code className="text-[13px]">stepfun/step-3.5-flash:free</code>)
              is the primary teacher — it generates the rewrite pairs and the
              six strategy variants per source that together make up the
              89,688-record training pool. <strong>Nemotron Nano 30B</strong>{" "}
              (<code className="text-[13px]">nvidia/nemotron-3-nano-30b-a3b:free</code>)
              is used separately as an independent binary sarcasm classifier,
              re-checking NHDSD source-headline labels where the original
              NHDSD label and Step 3.5 Flash disagreed.
            </p>
            <p>
              This is the standard strong-teacher → small-student distillation
              pattern that Stanford Alpaca, Vicuna, and Orca popularised. The
              teachers act as a <em>label factory</em> for a supervised
              training set that would otherwise need thousands of human-hours
              to produce; the small student then carries the task at inference
              time.
            </p>
          </>
        ),
        links: [
          { label: "Data pipeline", href: "/pipeline" },
          { label: "Training recipes", href: "/training" },
        ],
      },
      {
        num: "02",
        source: "instructor",
        question:
          "T5 is a simpler model and may not respond well to structured prompts — are you sure it's the right baseline?",
        answer: (
          <>
            <p className="mb-3">
              A fair concern going in, but the results flipped it: <strong>T5-Joint
              is the best model in our lineup</strong>, beating BART-Base,
              BART-CE, all BART-RL variants, and both LLaMA LoRA variants on
              human evaluation (43.6% strict success vs the runner-up at 32.1%).
            </p>
            <p className="mb-3">
              Two factors explain this. First, joint-task training (predict
              strategy + rewrite simultaneously from a single prompt) gave T5
              an auxiliary signal that regularised the decoder. Second, the
              encoder-decoder architecture turned out to be a better fit for
              headline-length span rewriting than either the decoder-only LLaMA
              or the RL-tuned BART.
            </p>
            <p>
              The structured-prompt concern was real for BART-CE, which uses a
              longer context-enhanced prompt format and ended up at 28.6%
              strict success — worse than the simpler BART-Base at 30.7%.
            </p>
          </>
        ),
        links: [
          { label: "Training recipes", href: "/training" },
          { label: "Dashboard", href: "/dashboard" },
        ],
      },
    ],
  },
  {
    title: "Evaluation",
    blurb: "Clarifying what counts as a successful rewrite.",
    items: [
      {
        num: "03",
        source: "mentor",
        question:
          "How do you define a better rewrite? What does “better” actually mean?",
        answer: (
          <>
            <p className="mb-3">
              A rewrite is &quot;better&quot; if it satisfies both conditions
              simultaneously:
            </p>
            <ul className="list-disc ml-5 space-y-1.5 mb-3">
              <li>
                <strong>Sarcasm removed</strong> — the output is no longer read
                as sarcastic by a human annotator.
              </li>
              <li>
                <strong>Meaning preserved</strong> — the underlying claim or
                event is the same as the input, not paraphrased into a
                different statement.
              </li>
            </ul>
            <p className="mb-3">
              Both criteria are judged by two independent human annotators on a
              140-sample gold set, with Cohen&apos;s κ ∈ [0.839, 0.884] across
              the three evaluated models. The headline metric we report is{" "}
              <strong>strict success rate</strong> — the fraction of samples
              where <em>both</em> annotators agreed the rewrite flipped AND
              preserved meaning.
            </p>
            <p>
              We also report seven automatic metrics (three classifier flip
              rates, semantic similarity, perplexity, BLEU vs input, edit
              distance, and an LLM-as-judge score), but we treat the human
              numbers as ground truth and the automatic metrics as diagnostic
              signals — see the classifier-vs-human disagreement story below.
            </p>
          </>
        ),
        links: [
          { label: "Evaluation methodology", href: "/eval" },
          { label: "Human evaluation results", href: "/human-eval" },
        ],
      },
    ],
  },
];

const SOURCE_LABEL: Record<FaqItem["source"], string> = {
  instructor: "Instructor",
  mentor: "Mentor",
};

export default function FaqPage() {
  return (
    <div className="min-h-screen px-4 md:px-12 py-12 md:py-20">
      <div className="max-w-3xl">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-4"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          FAQ · Responses to interim feedback
        </span>
        <h1
          className="text-[40px] md:text-[56px] leading-[1.05] tracking-[-0.8px] md:tracking-[-1.12px] text-foreground mb-4"
          style={{ fontFamily: "var(--font-dm-serif)" }}
        >
          Questions &amp; Answers
        </h1>
        <p className="text-[16px] md:text-[17px] leading-[1.6] text-foreground-secondary mb-3">
          This page collects the questions and comments raised during the
          CS4248 interim review by our instructor and project mentor, along
          with the team&apos;s responses.
        </p>
        <p className="text-[14px] leading-[1.6] text-muted mb-12">
          Additional questions from the poster roadshow and final presentation
          will be added here as they come in.
        </p>

        {SECTIONS.map((section) => (
          <section key={section.title} className="mb-14">
            <h2
              className="text-[24px] md:text-[28px] tracking-[-0.28px] text-foreground mb-2"
              style={{ fontFamily: "var(--font-dm-serif)" }}
            >
              {section.title}
            </h2>
            {section.blurb && (
              <p className="text-[14px] leading-[1.6] text-muted mb-6 max-w-2xl">
                {section.blurb}
              </p>
            )}
            <div className="space-y-8">
              {section.items.map((item) => (
                <div
                  key={item.num}
                  className="border border-border-card rounded-[18px] p-6 md:p-7"
                >
                  <div className="flex items-center gap-3 mb-3">
                    <span
                      className="text-[11px] tracking-[0.16px] text-muted font-code"
                      style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                    >
                      {item.num}
                    </span>
                    <span
                      className="text-[10px] tracking-[0.24px] uppercase px-2 py-0.5 rounded-full border border-border-card text-muted"
                      style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                    >
                      {SOURCE_LABEL[item.source]}
                    </span>
                  </div>
                  <h3
                    className="text-[18px] md:text-[19px] leading-[1.4] tracking-[-0.2px] text-foreground mb-4"
                    style={{ fontFamily: "var(--font-dm-serif)" }}
                  >
                    {item.question}
                  </h3>
                  <div className="text-[15px] leading-[1.65] text-foreground-secondary">
                    {item.answer}
                  </div>
                  {item.links && item.links.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-5 pt-4 border-t border-border-light">
                      {item.links.map((link) =>
                        link.external ? (
                          <a
                            key={link.href}
                            href={link.href}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center text-[12px] px-2.5 py-1 rounded-full border border-border-card text-foreground-secondary hover:text-accent-blue hover:border-accent-blue/30 transition-colors"
                          >
                            {link.label} ↗
                          </a>
                        ) : (
                          <Link
                            key={link.href}
                            href={link.href}
                            className="inline-flex items-center text-[12px] px-2.5 py-1 rounded-full border border-border-card text-foreground-secondary hover:text-accent-blue hover:border-accent-blue/30 transition-colors"
                          >
                            {link.label} →
                          </Link>
                        )
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>
        ))}

        <p
          className="text-[11px] text-muted/70 mt-4 tracking-[0.16px]"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          CS4248 / AY2025/26 S2 / Team 14 / NUS
        </p>
      </div>
    </div>
  );
}
