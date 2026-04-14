import Link from "next/link";

const REPO = "https://github.com/SeeYangZhi/Project-LLMao/blob/main";

type HyperRow = { key: string; value: string };

type ModelTraining = {
  key: string;
  display: string;
  base: string;
  highlight?: string;
  summary: string;
  script: { label: string; href: string };
  rows: HyperRow[];
};

const SFT_SEQ2SEQ: ModelTraining[] = [
  {
    key: "bart-base",
    display: "BART-Base",
    base: "facebook/bart-base (140M)",
    summary:
      "Plain supervised fine-tuning on the 10,868 headline→rewrite pairs from the main sar-to-non split. Input is the raw sarcastic headline with no prefix — BART is pretrained with its own denoising objective and doesn't expect a task token.",
    script: { label: "scripts/train.py", href: `${REPO}/scripts/train.py` },
    rows: [
      { key: "Train size", value: "10,868 pairs" },
      { key: "Val size", value: "1,356 pairs" },
      { key: "Epochs", value: "5 (early stop, patience 2)" },
      { key: "Batch size", value: "16" },
      { key: "Learning rate", value: "3e-4" },
      { key: "Max length", value: "128 tokens" },
      { key: "Warmup steps", value: "500" },
      { key: "Weight decay", value: "0.01" },
      { key: "Best metric", value: "BLEU on val" },
      { key: "Precision", value: "bf16 (if CUDA)" },
    ],
  },
  {
    key: "bart-ce",
    display: "BART-CE",
    base: "facebook/bart-base (140M)",
    highlight: "Context-enhanced training data",
    summary:
      "Same BART architecture and hyperparameters as BART-Base, but trained on the context-enhanced data split: each training pair is conditioned on the scraped article body in addition to the headline. Smaller split (8,258 pairs) because not every headline has a scrape-able article body.",
    script: { label: "scripts/train.py", href: `${REPO}/scripts/train.py` },
    rows: [
      { key: "Train size", value: "8,258 pairs (with body)" },
      { key: "Val size", value: "1,029 pairs" },
      { key: "Data split", value: "sar_to_non_context_enhanced" },
      { key: "Epochs", value: "5 (early stop, patience 2)" },
      { key: "Batch size", value: "16" },
      { key: "Learning rate", value: "3e-4" },
      { key: "Max length", value: "128 tokens" },
      { key: "Best metric", value: "BLEU on val" },
    ],
  },
  {
    key: "t5-control",
    display: "T5-Control",
    base: "google-t5/t5-small (60M)",
    summary:
      "T5 baseline with no strategy information — the model has to figure out what's sarcastic and how to fix it in a single step. Input is prefixed with the task token `desarcasm:` because T5 was pretrained with task prefixes.",
    script: { label: "scripts/train.py", href: `${REPO}/scripts/train.py` },
    rows: [
      { key: "Input format", value: '"desarcasm: {headline}"' },
      { key: "Train size", value: "10,868 pairs" },
      { key: "Epochs", value: "5 (early stop, patience 2)" },
      { key: "Batch size", value: "16" },
      { key: "Learning rate", value: "3e-4" },
      { key: "Max length", value: "128 tokens" },
    ],
  },
  {
    key: "t5-joint",
    display: "T5-Joint",
    base: "google/t5-base (220M)",
    highlight: "Best model by human eval",
    summary:
      "T5-base trained to do two things at once: classify the sarcasm strategy AND rewrite the headline. The target string has the format `strategy: <type> rewrite: <headline>`. The strategy prefix forces task decomposition before generation — identify what's sarcastic first, then remove it — and this is why T5-Joint wins on meaning preservation (16.4% meaning change vs T5-Control's 25%).",
    script: { label: "scripts/train.py", href: `${REPO}/scripts/train.py` },
    rows: [
      { key: "Output format", value: '"strategy: X rewrite: Y"' },
      { key: "Train size", value: "10,868 strategy-labeled pairs" },
      { key: "Epochs", value: "5 (early stop, patience 2)" },
      { key: "Batch size", value: "16" },
      { key: "Learning rate", value: "3e-4" },
      { key: "Max length", value: "128 tokens" },
      { key: "Why it wins", value: "Decomposition before generation" },
    ],
  },
  {
    key: "t5-joint-small",
    display: "T5-Joint (small)",
    base: "google-t5/t5-small (60M)",
    summary:
      "Older T5-small variant of the joint model — same training recipe, smaller backbone. Kept for comparison to show that T5-Joint's edge comes from the strategy-prefix trick, not from model size.",
    script: { label: "scripts/train.py", href: `${REPO}/scripts/train.py` },
    rows: [
      { key: "Output format", value: '"strategy: X rewrite: Y"' },
      { key: "Epochs", value: "5 (early stop, patience 2)" },
      { key: "Batch size", value: "16" },
      { key: "Learning rate", value: "3e-4" },
    ],
  },
];

const RL_MODELS: ModelTraining[] = [
  {
    key: "bart-rl",
    display: "BART-RL",
    base: "BART-Base SFT checkpoint",
    summary:
      "Takes the BART-Base SFT checkpoint as both policy and frozen reference. Generates outputs via sampling, scores them with a sarcasm classifier, and updates the policy with REINFORCE + a KL penalty to stop it drifting. The reward is a weighted sum of style (classifier) and content preservation (ROUGE-L against the reference).",
    script: { label: "scripts/train_rl.py", href: `${REPO}/scripts/train_rl.py` },
    rows: [
      { key: "Policy init", value: "BART-Base SFT checkpoint" },
      { key: "Reference", value: "Same checkpoint, frozen" },
      { key: "Reward (style)", value: "1 − P(sarcastic)" },
      { key: "Reward (content)", value: "ROUGE-L vs reference" },
      { key: "Reward blend", value: "α·style + (1−α)·content, α = 0.5" },
      { key: "KL coeff", value: "0.2" },
      { key: "Learning rate", value: "1e-5 (low — policy drift risk)" },
      { key: "Epochs", value: "3" },
      { key: "Gradient clip", value: "1.0" },
      { key: "Baseline", value: "EMA (0.9 decay)" },
    ],
  },
  {
    key: "bart-ce-rl",
    display: "BART-CE+RL",
    base: "BART-CE SFT checkpoint",
    summary:
      "Same RL recipe as BART-RL, but starts from the context-enhanced SFT checkpoint instead of the plain one. Intended to stack the benefits of article-aware supervision with the reward-driven polish.",
    script: { label: "scripts/train_rl.py", href: `${REPO}/scripts/train_rl.py` },
    rows: [
      { key: "Policy init", value: "BART-CE SFT checkpoint" },
      { key: "Reward formula", value: "Same as BART-RL" },
      { key: "KL coeff", value: "0.2" },
      { key: "Learning rate", value: "1e-5" },
      { key: "Epochs", value: "3" },
    ],
  },
];

const LLAMA_MODELS: ModelTraining[] = [
  {
    key: "llama-base",
    display: "LLaMA 3.2 1B",
    base: "meta-llama/Llama-3.2-1B-Instruct",
    summary:
      "Instruction-tuned LLaMA fine-tuned via LoRA adapters — only ~6M of the 1.24B parameters are trained. Uses the chat template with a system prompt, loss is masked to the assistant response only (the prompt tokens are set to −100). The SFT checkpoint is then merged back into the base model and exported to GGUF so LMStudio can serve it on consumer hardware.",
    script: { label: "scripts/train_llama.py", href: `${REPO}/scripts/train_llama.py` },
    rows: [
      { key: "LoRA rank", value: "r = 16" },
      { key: "LoRA alpha", value: "32" },
      { key: "LoRA dropout", value: "0.05" },
      { key: "Target modules", value: "q, k, v, o, gate, up, down" },
      { key: "Trainable params", value: "~6M of 1.24B (0.5%)" },
      { key: "Learning rate", value: "2e-4" },
      { key: "Batch × grad accum", value: "8 × 2 = 16 effective" },
      { key: "Epochs", value: "3" },
      { key: "Max length", value: "256 tokens" },
      { key: "Scheduler", value: "Cosine, 5% warmup" },
      { key: "Precision", value: "bf16 + gradient checkpointing" },
    ],
  },
  {
    key: "llama-context",
    display: "LLaMA 3.2 1B (context)",
    base: "meta-llama/Llama-3.2-1B-Instruct",
    highlight: "Scraped article body in the prompt",
    summary:
      "Same LoRA recipe as the base LLaMA variant, but the user message includes the scraped article body alongside the headline. This tests whether the model can use the article to ground its rewrite in the real event. Bumping max_length to 1024 to fit the article forces a smaller batch size.",
    script: {
      label: "scripts/train_llama_context.py",
      href: `${REPO}/scripts/train_llama_context.py`,
    },
    rows: [
      { key: "LoRA config", value: "Same as base (r=16, α=32)" },
      { key: "Learning rate", value: "2e-4" },
      { key: "Batch × grad accum", value: "4 × 4 = 16 effective" },
      { key: "Epochs", value: "3" },
      { key: "Max length", value: "1024 tokens (to fit article body)" },
      {
        key: "Article cache",
        value: "data/processed/intermediate/article_scrape_cache.jsonl",
      },
      { key: "Prompt format", value: "Headline + 'Article context:' + body" },
    ],
  },
];

const LLAMA_SYSTEM_PROMPT =
  "You are a writing assistant. Rewrite sarcastic news headlines as neutral, factual equivalents that preserve the core meaning without irony or mockery. Respond with only the rewritten headline, no explanation.";

const SUBTYPES = [
  "sarcasm",
  "irony",
  "satire",
  "overstatement",
  "understatement",
  "rhetorical_question",
];

function ModelCard({ m }: { m: ModelTraining }) {
  return (
    <article
      id={m.key}
      className="border border-border-card rounded-[22px] p-5 md:p-7"
    >
      <header className="mb-4 md:mb-5">
        <div className="flex items-center gap-3 mb-2 flex-wrap">
          <h3 className="text-[20px] md:text-[22px] tracking-[-0.22px] text-foreground">
            {m.display}
          </h3>
          {m.highlight && (
            <span
              className="text-[10px] tracking-[0.16px] uppercase text-accent-blue px-2 py-0.5 rounded-full bg-accent-blue/10"
              style={{ fontFamily: "var(--font-jetbrains-mono)" }}
            >
              {m.highlight}
            </span>
          )}
        </div>
        <p
          className="text-[12px] md:text-[13px] text-muted"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          {m.base}
        </p>
      </header>

      <p className="text-[14px] md:text-[15px] leading-[1.6] text-foreground-secondary mb-5 max-w-2xl">
        {m.summary}
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1.5 mb-4">
        {m.rows.map((r) => (
          <div
            key={r.key}
            className="flex items-baseline justify-between gap-3 py-1 border-b border-border-card/60 text-[12px] md:text-[13px]"
          >
            <span
              className="text-muted"
              style={{ fontFamily: "var(--font-jetbrains-mono)" }}
            >
              {r.key}
            </span>
            <span className="text-foreground-secondary tabular-nums text-right">
              {r.value}
            </span>
          </div>
        ))}
      </div>

      <a
        href={m.script.href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-[12px] md:text-[13px] text-accent-blue hover:underline"
        style={{ fontFamily: "var(--font-jetbrains-mono)" }}
      >
        {m.script.label} ↗
      </a>
    </article>
  );
}

export default function TrainingPage() {
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
          Model Training
        </h1>
        <p className="text-[16px] md:text-[18px] leading-[1.55] text-foreground-secondary max-w-3xl">
          Every model in the dashboard traces back to one of four training
          recipes. This page is the exact hyperparameters, data splits, and
          loss formulations we used — read it alongside the{" "}
          <Link href="/pipeline" className="text-accent-blue hover:underline">
            data pipeline
          </Link>{" "}
          and the{" "}
          <Link href="/eval" className="text-accent-blue hover:underline">
            evaluation methodology
          </Link>
          .
        </p>
      </section>

      {/* Overview */}
      <section className="px-4 md:px-12 pb-8 md:pb-10">
        <div className="border border-border-card rounded-[22px] p-5 md:p-6 max-w-4xl">
          <h3 className="text-[16px] md:text-[18px] tracking-[-0.18px] text-foreground mb-4">
            Four training recipes
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 md:gap-4">
            {[
              {
                tag: "01",
                title: "Supervised Fine-Tuning (seq2seq)",
                body: "BART and T5 variants trained with cross-entropy on 10,868 sarcastic→non-sarcastic headline pairs. 5 epochs, early stopping on val BLEU.",
                models: 5,
              },
              {
                tag: "02",
                title: "Reinforcement Learning (REINFORCE + KL)",
                body: "Takes an SFT BART checkpoint as policy, uses a sarcasm classifier + ROUGE-L as reward, KL penalty against the frozen reference.",
                models: 2,
              },
              {
                tag: "03",
                title: "LoRA Instruction Tuning",
                body: "LLaMA 3.2 1B fine-tuned via low-rank adapters on 7 projection layers. Loss masked to the assistant response only.",
                models: 2,
              },
              {
                tag: "04",
                title: "Ablation study",
                body: "Six T5-Joint retrains, each with one of the six sarcasm subtypes held out of the training data to measure its contribution.",
                models: 6,
              },
            ].map((item) => (
              <div
                key={item.tag}
                className="border border-border-card rounded-xl p-4"
              >
                <div className="flex items-baseline gap-2 mb-1.5">
                  <span
                    className="text-[10px] text-muted"
                    style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                  >
                    {item.tag}
                  </span>
                  <h4 className="text-[14px] md:text-[15px] text-foreground">
                    {item.title}
                  </h4>
                </div>
                <p className="text-[12px] md:text-[13px] leading-[1.5] text-muted">
                  {item.body}
                </p>
                <div
                  className="text-[10px] text-accent-blue mt-2 tracking-[0.16px] uppercase"
                  style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                >
                  {item.models} models
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Section 1: SFT seq2seq */}
      <section className="px-4 md:px-12 pb-10 md:pb-12">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Section 01
        </span>
        <h2
          className="text-[24px] md:text-[32px] leading-[1.05] tracking-[-0.32px] md:tracking-[-0.48px] text-foreground mb-3"
          style={{ fontFamily: "var(--font-dm-serif)" }}
        >
          Supervised Fine-Tuning
        </h2>
        <p className="text-[14px] md:text-[16px] leading-[1.6] text-foreground-secondary max-w-3xl mb-6 md:mb-8">
          Standard cross-entropy training with the HuggingFace{" "}
          <code className="text-[13px] text-accent-purple">Seq2SeqTrainer</code>,
          early stopping on validation BLEU (patience 2), and a cosine warmup
          schedule. Every seq2seq model below uses the same trainer — the
          differences are the base checkpoint and whether the input carries a
          strategy prefix.
        </p>
        <div className="space-y-4 md:space-y-5 max-w-4xl">
          {SFT_SEQ2SEQ.map((m) => (
            <ModelCard key={m.key} m={m} />
          ))}
        </div>
      </section>

      {/* Section 2: RL */}
      <section className="px-4 md:px-12 pb-10 md:pb-12">
        <div className="border-t border-border-light pt-10 md:pt-12 max-w-4xl">
          <span
            className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3"
            style={{ fontFamily: "var(--font-jetbrains-mono)" }}
          >
            Section 02
          </span>
          <h2
            className="text-[24px] md:text-[32px] leading-[1.05] tracking-[-0.32px] md:tracking-[-0.48px] text-foreground mb-3"
            style={{ fontFamily: "var(--font-dm-serif)" }}
          >
            Reinforcement Learning
          </h2>
          <p className="text-[14px] md:text-[16px] leading-[1.6] text-foreground-secondary mb-5">
            REINFORCE with a KL penalty against the frozen SFT reference — the
            recipe ViSP{" "}
            <a
              href="https://arxiv.org/abs/2507.09482"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent-blue hover:underline"
            >
              (arxiv 2507.09482)
            </a>{" "}
            used for sarcasm <em>generation</em>, inverted here for sarcasm{" "}
            <em>removal</em>. Pure style reward saturates instantly because the
            SFT outputs already score ~1.0 on the classifier, so we blend in a
            ROUGE-L content-preservation term to keep the policy from
            collapsing to &ldquo;delete everything&rdquo;.
          </p>

          {/* Reward + loss formula */}
          <div className="border border-accent-purple/30 bg-accent-purple/[0.03] rounded-[22px] p-5 md:p-6 mb-6 md:mb-8">
            <h4
              className="text-[11px] tracking-[0.28px] uppercase text-accent-purple mb-3"
              style={{ fontFamily: "var(--font-jetbrains-mono)" }}
            >
              Loss formulation
            </h4>
            <dl className="text-[13px] md:text-[14px] space-y-3">
              <div>
                <dt className="text-muted mb-0.5">Reward</dt>
                <dd
                  className="text-foreground tabular-nums"
                  style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                >
                  r = α · (1 − P<sub>sarcastic</sub>(output)) + (1 − α) ·
                  ROUGE-L(output, ref), α = 0.5
                </dd>
              </div>
              <div>
                <dt className="text-muted mb-0.5">REINFORCE loss</dt>
                <dd
                  className="text-foreground"
                  style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                >
                  L<sub>rl</sub> = −(r − baseline) · Σ log π<sub>θ</sub>(y | x)
                </dd>
              </div>
              <div>
                <dt className="text-muted mb-0.5">Total loss</dt>
                <dd
                  className="text-foreground"
                  style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                >
                  L = L<sub>rl</sub> + β · KL(π<sub>θ</sub> ‖ π<sub>ref</sub>),
                  β = 0.2
                </dd>
              </div>
              <div>
                <dt className="text-muted mb-0.5">Baseline</dt>
                <dd className="text-foreground-secondary">
                  Exponential moving average of batch reward (decay 0.9)
                </dd>
              </div>
            </dl>
          </div>

          <div className="space-y-4 md:space-y-5">
            {RL_MODELS.map((m) => (
              <ModelCard key={m.key} m={m} />
            ))}
          </div>

          <div className="mt-5 md:mt-6 border border-red-200 bg-red-50/50 rounded-[22px] p-5 md:p-6">
            <h4 className="text-[14px] md:text-[15px] text-red-700 mb-2">
              Known failure mode: reward hacking
            </h4>
            <p className="text-[13px] md:text-[14px] leading-[1.6] text-red-800/80">
              Human eval shows BART-RL has a 40.7% meaning-change rate — more
              than double T5-Joint&apos;s 16.4%. The model learned that
              deleting sarcastic tokens reduces P(sarcastic) while still
              preserving enough ROUGE-L overlap to satisfy the content reward.
              Deletion optimizes the composite reward without actually
              rewriting. We document this in detail on the{" "}
              <Link href="/eval" className="underline">
                eval page
              </Link>
              .
            </p>
          </div>
        </div>
      </section>

      {/* Section 3: LLaMA LoRA */}
      <section className="px-4 md:px-12 pb-10 md:pb-12">
        <div className="border-t border-border-light pt-10 md:pt-12 max-w-4xl">
          <span
            className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3"
            style={{ fontFamily: "var(--font-jetbrains-mono)" }}
          >
            Section 03
          </span>
          <h2
            className="text-[24px] md:text-[32px] leading-[1.05] tracking-[-0.32px] md:tracking-[-0.48px] text-foreground mb-3"
            style={{ fontFamily: "var(--font-dm-serif)" }}
          >
            LoRA Instruction Tuning
          </h2>
          <p className="text-[14px] md:text-[16px] leading-[1.6] text-foreground-secondary mb-6">
            LLaMA 3.2 1B is a decoder-only chat model, so the recipe looks
            nothing like the seq2seq pipeline. We use PEFT LoRA adapters on
            every attention + MLP projection, keeping the base weights frozen,
            and mask the loss to the assistant response. The system prompt and
            user message are encoded with the Llama 3 chat template so the
            model stays aligned with its instruction-tuned prior.
          </p>

          {/* System prompt card */}
          <div className="border border-border-card rounded-[22px] p-5 md:p-6 mb-6 md:mb-8">
            <h4
              className="text-[11px] tracking-[0.28px] uppercase text-muted mb-3"
              style={{ fontFamily: "var(--font-jetbrains-mono)" }}
            >
              System prompt
            </h4>
            <p
              className="text-[13px] md:text-[14px] leading-[1.65] text-foreground-secondary italic"
              style={{ fontFamily: "var(--font-jetbrains-mono)" }}
            >
              &ldquo;{LLAMA_SYSTEM_PROMPT}&rdquo;
            </p>
          </div>

          <div className="space-y-4 md:space-y-5">
            {LLAMA_MODELS.map((m) => (
              <ModelCard key={m.key} m={m} />
            ))}
          </div>
        </div>
      </section>

      {/* Section 4: Ablation */}
      <section className="px-4 md:px-12 pb-12 md:pb-16">
        <div className="border-t border-border-light pt-10 md:pt-12 max-w-4xl">
          <span
            className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3"
            style={{ fontFamily: "var(--font-jetbrains-mono)" }}
          >
            Section 04
          </span>
          <h2
            className="text-[24px] md:text-[32px] leading-[1.05] tracking-[-0.32px] md:tracking-[-0.48px] text-foreground mb-3"
            style={{ fontFamily: "var(--font-dm-serif)" }}
          >
            Ablation Study
          </h2>
          <p className="text-[14px] md:text-[16px] leading-[1.6] text-foreground-secondary mb-5 md:mb-6">
            Six retrainings of the T5-Joint recipe, each with one sarcasm
            subtype dropped from the training data. The goal: measure whether
            any single subtype is load-bearing for the joint model&apos;s
            performance. The finding on the{" "}
            <Link href="/dashboard" className="text-accent-blue hover:underline">
              dashboard
            </Link>
            : ablation models cluster within{" "}
            <span className="tabular-nums">0.005</span> of each other on
            similarity — the model learns generic sarcasm patterns that
            transfer across subtypes.
          </p>

          <div className="border border-border-card rounded-[22px] p-5 md:p-6 mb-5">
            <h4 className="text-[14px] md:text-[15px] text-foreground mb-3">
              Shared training recipe
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1.5 text-[12px] md:text-[13px]">
              {[
                ["Base", "google/t5-base (220M)"],
                ["Output format", "strategy: X rewrite: Y"],
                ["Epochs", "5 (early stop, patience 2)"],
                ["Batch size", "16"],
                ["Learning rate", "3e-4"],
                ["Max length", "128 tokens"],
              ].map(([k, v]) => (
                <div
                  key={k}
                  className="flex items-baseline justify-between gap-3 py-1 border-b border-border-card/60"
                >
                  <span
                    className="text-muted"
                    style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                  >
                    {k}
                  </span>
                  <span className="text-foreground-secondary">{v}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 md:gap-4">
            {SUBTYPES.map((s) => (
              <div
                key={s}
                className="border border-border-card rounded-xl p-4"
              >
                <div
                  className="text-[10px] tracking-[0.16px] uppercase text-muted mb-1"
                  style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                >
                  Held out
                </div>
                <div className="text-[13px] md:text-[14px] text-foreground">
                  {s.replace(/_/g, " ")}
                </div>
                <div className="text-[11px] text-muted mt-2">
                  ablation_without_{s}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
