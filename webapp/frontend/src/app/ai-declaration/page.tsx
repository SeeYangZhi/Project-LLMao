import Link from "next/link";

type ToolRow = {
  tool: string;
  purpose: string;
  usage: string;
};

const DECLARED_TOOLS: ToolRow[] = [
  {
    tool: "Step 3.5 Flash (stepfun/step-3.5-flash:free, via OpenRouter)",
    purpose:
      "Primary teacher for the training corpus. Generated the non-sarcastic → sarcastic rewrite pairs from NHDSD source headlines, and then in a second pass produced the five missing strategy variants per source (sarcasm, irony, satire, understatement, overstatement, rhetorical_question).",
    usage:
      "The two Step 3.5 Flash passes together produce the 89,688-record training pool (14,948 sources × 6 strategies). Used as-is for training — no downstream LLM filter was applied to these pairs.",
  },
  {
    tool: "Nemotron Nano 30B (nvidia/nemotron-3-nano-30b-a3b:free, via OpenRouter)",
    purpose:
      "Independent binary sarcasm classifier used to cross-validate source-headline labels in NHDSD where the original NHDSD label and Step 3.5 Flash disagreed.",
    usage:
      "Ran only on the disagreement subset as a tiebreaker to estimate NHDSD mislabel rate. This is a QA pass on source headline labels upstream of pair generation — it does not filter or re-annotate the 89,688 training pairs themselves.",
  },
  {
    tool: "Google Gemini 2.5 Flash (via API)",
    purpose:
      "Acted as one of the seven evaluation signals — the LLM-as-judge score reported in the dashboard — rating whether each model output preserves meaning while removing sarcasm.",
    usage:
      "Used as an automated metric alongside six non-LLM metrics. Human evaluation (140 samples × 3 models × 2 annotators, κ > 0.8) is the primary ground truth, not the Gemini score. See /eval.",
  },
  {
    tool: "Anthropic Claude (Claude Code, Opus / Sonnet)",
    purpose:
      "Pair-programming assistant for implementing the webapp (Next.js frontend, FastAPI backend), refactoring training scripts, and drafting documentation.",
    usage:
      "All generated code was read, edited, run, and debugged by team members before being committed. Claude did not make architectural decisions autonomously — every recipe, metric, and experiment was specified by the team.",
  },
  {
    tool: "GitHub Copilot",
    purpose: "Inline autocomplete during routine coding (loops, boilerplate, type signatures).",
    usage: "Suggestions were accepted or rejected line-by-line by the author.",
  },
  {
    tool: "ChatGPT + Codex",
    purpose: "Generating training and other code. Asking ML engineering related tasks", 
    usage: "Output reviewed manually before use. AI did not suggest what experiment to run, human set research directions and directive on what to do and the expected end output, e.g. I want a customisable SLURM training script.",
  },
];

const NOT_USED_FOR = [
  "Formulating the research question, hypotheses, or experimental design.",
  "Selecting the 14 models, four training recipes, or the seven-metric evaluation pipeline.",
  "Running training jobs or generating model outputs on the held-out test set.",
  "Manually labelling the 140-sample gold human-evaluation set (done by two team members independently).",
  "Drawing conclusions from results or deciding which findings to report.",
];

export default function AIDeclarationPage() {
  return (
    <div className="min-h-screen px-4 md:px-12 py-12 md:py-20">
      <div className="max-w-3xl">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-4"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Appendix · Academic Integrity
        </span>
        <h1
          className="text-[40px] md:text-[56px] leading-[1.05] tracking-[-0.8px] md:tracking-[-1.12px] text-foreground mb-4"
          style={{ fontFamily: "var(--font-dm-serif)" }}
        >
          Declaration on the Use of AI
        </h1>
        <p className="text-[16px] md:text-[17px] leading-[1.6] text-foreground-secondary mb-3">
          This page is the team&apos;s acknowledgement of generative AI tools
          used in Project LLMao, filed in the spirit of Section 4.3 of the{" "}
          <a
            href="https://ctlt.nus.edu.sg/wp-content/uploads/2025/11/Policy-for-Use-of-AI-in-Teaching-and-Learning-2024.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent-blue underline underline-offset-2 hover:text-accent-blue/80"
          >
            NUS Policy for Use of AI in Teaching and Learning (7 Aug 2024)
          </a>
          .
        </p>
        <p className="text-[15px] leading-[1.6] text-muted mb-10">
          The project itself is a study of small language models. Generative AI
          is both an <em>object of study</em> (the 14 fine-tuned models) and a{" "}
          <em>tool we used</em> during data preparation, evaluation, and
          engineering. We separate the two below.
        </p>

        {/* Declaration table */}
        <section className="mb-12">
          <h2
            className="text-[22px] md:text-[24px] tracking-[-0.24px] text-foreground mb-4"
            style={{ fontFamily: "var(--font-dm-serif)" }}
          >
            AI tools used, and how
          </h2>
          <div className="border border-border-card rounded-[18px] overflow-hidden">
            <div className="hidden md:grid grid-cols-[1.1fr_1.4fr_1.5fr] gap-0 bg-foreground/[0.03] border-b border-border-card">
              <div
                className="px-5 py-3 text-[11px] tracking-[0.28px] uppercase text-muted"
                style={{ fontFamily: "var(--font-jetbrains-mono)" }}
              >
                AI Tool
              </div>
              <div
                className="px-5 py-3 text-[11px] tracking-[0.28px] uppercase text-muted"
                style={{ fontFamily: "var(--font-jetbrains-mono)" }}
              >
                Purpose
              </div>
              <div
                className="px-5 py-3 text-[11px] tracking-[0.28px] uppercase text-muted"
                style={{ fontFamily: "var(--font-jetbrains-mono)" }}
              >
                How the output was used
              </div>
            </div>
            {DECLARED_TOOLS.map((row, idx) => (
              <div
                key={row.tool}
                className={`md:grid md:grid-cols-[1.1fr_1.4fr_1.5fr] gap-0 ${
                  idx < DECLARED_TOOLS.length - 1
                    ? "border-b border-border-card"
                    : ""
                }`}
              >
                <div className="px-5 pt-5 pb-2 md:py-5 text-[14px] leading-[1.55] text-foreground">
                  {row.tool}
                </div>
                <div className="px-5 pb-2 md:py-5 text-[14px] leading-[1.6] text-foreground-secondary">
                  {row.purpose}
                </div>
                <div className="px-5 pb-5 md:py-5 text-[14px] leading-[1.6] text-muted">
                  {row.usage}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Not used for */}
        <section className="mb-12">
          <h2
            className="text-[22px] md:text-[24px] tracking-[-0.24px] text-foreground mb-4"
            style={{ fontFamily: "var(--font-dm-serif)" }}
          >
            What AI was <em>not</em> used for
          </h2>
          <ul className="space-y-2">
            {NOT_USED_FOR.map((item) => (
              <li
                key={item}
                className="text-[15px] leading-[1.6] text-foreground-secondary pl-5 relative"
              >
                <span className="absolute left-0 top-[0.55em] w-2 h-px bg-muted" />
                {item}
              </li>
            ))}
          </ul>
        </section>

        {/* Responsibility */}
        <section className="mb-12 border-t border-border-light pt-10">
          <h2
            className="text-[22px] md:text-[24px] tracking-[-0.24px] text-foreground mb-4"
            style={{ fontFamily: "var(--font-dm-serif)" }}
          >
            Responsibility
          </h2>
          <p className="text-[15px] leading-[1.65] text-foreground-secondary mb-3">
            Team 14 is solely responsible for the content of this report, the
            webapp, the code, the experimental results, and any errors therein.
            Every AI-assisted output — whether a generated training label, a
            code suggestion, or a proofreading pass — was reviewed by a team
            member before being integrated into the final submission.
          </p>
          <p className="text-[15px] leading-[1.65] text-muted">
            We have not used AI tools to generate this declaration&apos;s
            substantive content about what the team did or did not do; those
            statements are authored by the team. Phrasing and formatting passes
            were AI-assisted and then edited by hand.
          </p>
        </section>

        {/* Project context crosslinks */}
        <section className="border-t border-border-light pt-8">
          <span
            className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-4"
            style={{ fontFamily: "var(--font-jetbrains-mono)" }}
          >
            Further reading in this report
          </span>
          <div className="flex flex-wrap gap-2">
            {[
              { href: "/pipeline", label: "Data pipeline (LLM annotation)" },
              { href: "/training", label: "Model training recipes" },
              { href: "/eval", label: "Evaluation methodology" },
              { href: "/human-eval", label: "Human evaluation" },
            ].map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="inline-flex items-center text-[13px] px-3 py-1.5 rounded-full border border-border-card text-foreground-secondary hover:text-accent-blue hover:border-accent-blue/30 transition-colors"
              >
                {link.label} →
              </Link>
            ))}
          </div>
        </section>

        <p className="text-[11px] text-muted/70 mt-12 tracking-[0.16px]" style={{ fontFamily: "var(--font-jetbrains-mono)" }}>
          CS4248 / AY2025/26 S2 / Team 14 / NUS
        </p>
      </div>
    </div>
  );
}
