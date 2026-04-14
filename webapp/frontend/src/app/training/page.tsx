import Link from "next/link";

const REPO = "https://github.com/SeeYangZhi/Project-LLMao/blob/main";
const T5_REPO = "https://github.com/camille-readbean/CS4248-project-AY2526S2/blob/main";

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
];

// T5 models live in a separate repo with a different training recipe.
// Source: github.com/camille-readbean/CS4248-project-AY2526S2
//   scripts/finetune_T5.py
//   scripts/slurm_finetune_t5.sh → slurm_finetune_t5.py
//   scripts/prepare_t5_datasets.py
const T5_MODELS: ModelTraining[] = [
  {
    key: "t5-joint",
    display: "T5-Joint",
    base: "google-t5/t5-base (220M)",
    highlight: "Best model by human eval",
    summary:
      "T5-base trained to do two things at once on every example: classify the sarcasm strategy AND rewrite the headline. The input is prefixed with 'rewrite to non-sarcastic and predict strategy: ' and the target is 'strategy: <type> rewrite: <headline>'. Forcing the model to emit the strategy token before the rewrite makes it decompose the task (identify what's sarcastic first, then remove it) and is why T5-Joint wins human eval on meaning preservation — 16.4% meaning change vs T5-Control's 25%.",
    script: {
      label: "camille-readbean/scripts/finetune_T5.py",
      href: `${T5_REPO}/scripts/finetune_T5.py`,
    },
    rows: [
      {
        key: "Input prefix",
        value: '"rewrite to non-sarcastic and predict strategy: "',
      },
      { key: "Target format", value: '"strategy: {strategy} rewrite: {rewrite}"' },
      { key: "Data split", value: "data/joint_and_ablate_prepared/joint (80/10/10 stratified)" },
      { key: "Epochs", value: "4" },
      { key: "Per-device batch", value: "8" },
      { key: "Grad accum", value: "2 (effective batch 16)" },
      { key: "Learning rate", value: "3e-4" },
      { key: "Scheduler", value: "Cosine, 6% warmup" },
      { key: "Weight decay", value: "0.01" },
      { key: "Max source/target len", value: "1248 tokens" },
      { key: "Best metric", value: "eval_loss (predict_with_generate=True)" },
      { key: "Precision", value: "fp16" },
      { key: "Compute", value: "1× NV GPU, 32G mem, SLURM gpu-long" },
    ],
  },
  {
    key: "t5-control",
    display: "T5-Control",
    base: "google-t5/t5-base (220M)",
    summary:
      "Same T5 recipe and split as T5-Joint but the strategy token is stripped from both input and output — the model only sees 'rewrite to non-sarcastic: {sarcastic}' and outputs the plain rewrite. This isolates the contribution of the strategy-prefix trick: any difference on meaning preservation between T5-Joint and T5-Control is attributable to the joint objective, not data or backbone.",
    script: {
      label: "camille-readbean/scripts/slurm_finetune_t5_control.sh",
      href: `${T5_REPO}/scripts/slurm_finetune_t5_control.sh`,
    },
    rows: [
      { key: "Input prefix", value: '"rewrite to non-sarcastic: "' },
      { key: "Target format", value: "Plain rewrite (no strategy token)" },
      { key: "Data split", value: "data/joint_and_ablate_prepared/control (same as joint)" },
      { key: "Epochs", value: "4" },
      { key: "Per-device batch", value: "8" },
      { key: "Grad accum", value: "2 (effective batch 16)" },
      { key: "Learning rate", value: "3e-4" },
      { key: "Scheduler", value: "Cosine, 6% warmup" },
      { key: "Max source/target len", value: "1248 tokens" },
      { key: "Precision", value: "fp16" },
    ],
  },
  {
    key: "t5-joint-small",
    display: "T5-Joint (small)",
    base: "google-t5/t5-small (60M)",
    summary:
      "Earlier T5-small variant of the joint model using the default model flag in Camille's slurm script. Same training recipe as T5-Joint, smaller backbone — kept in the evaluation to show that the joint model's edge comes from the strategy prefix, not just capacity. Still listed in the dashboard under the original name 'joint'.",
    script: {
      label: "camille-readbean/scripts/finetune_T5.py",
      href: `${T5_REPO}/scripts/finetune_T5.py`,
    },
    rows: [
      { key: "Input prefix", value: '"rewrite to non-sarcastic and predict strategy: "' },
      { key: "Target format", value: '"strategy: {strategy} rewrite: {rewrite}"' },
      { key: "Backbone", value: "t5-small (60M params)" },
      { key: "Everything else", value: "Identical to T5-Joint" },
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
        <p className="text-[13px] md:text-[14px] leading-[1.6] text-muted max-w-3xl mt-4">
          Note: the T5 family was trained on a separate stratified split with
          its own pipeline (Camille&apos;s{" "}
          <a
            href="https://github.com/camille-readbean/CS4248-project-AY2526S2"
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent-blue hover:underline"
          >
            CS4248-project-AY2526S2
          </a>{" "}
          repo). Same epochs and LR as the BART pipeline but different batch
          shape, max length, and best-metric criterion.
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
                tag: "01a",
                title: "BART SFT (Yang Zhi)",
                body: "BART-Base and BART-CE trained with cross-entropy on the sar-to-non splits. 5 epochs, early stopping on val BLEU, HuggingFace Seq2SeqTrainer.",
                models: 2,
              },
              {
                tag: "01b",
                title: "T5 SFT (Camille)",
                body: "T5-Joint, T5-Control, and the 6 ablations — 4 epochs, effective batch 16, max length 1248, fp16, eval_loss as best metric, SLURM orchestration.",
                models: 9,
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

      {/* Section 1a: BART SFT */}
      <section className="px-4 md:px-12 pb-10 md:pb-12">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Section 01a
        </span>
        <h2
          className="text-[24px] md:text-[32px] leading-[1.05] tracking-[-0.32px] md:tracking-[-0.48px] text-foreground mb-3"
          style={{ fontFamily: "var(--font-dm-serif)" }}
        >
          BART Supervised Fine-Tuning
        </h2>
        <p className="text-[14px] md:text-[16px] leading-[1.6] text-foreground-secondary max-w-3xl mb-6 md:mb-8">
          Yang Zhi&apos;s BART pipeline — cross-entropy training with the
          HuggingFace{" "}
          <code className="text-[13px] text-accent-purple">Seq2SeqTrainer</code>
          , early stopping on validation BLEU (patience 2), and a cosine warmup
          schedule. BART doesn&apos;t need a task prefix: it&apos;s pretrained
          with its own denoising objective, so the input is the raw sarcastic
          headline.
        </p>
        <div className="space-y-4 md:space-y-5 max-w-4xl">
          {SFT_SEQ2SEQ.map((m) => (
            <ModelCard key={m.key} m={m} />
          ))}
        </div>
      </section>

      {/* Section 1b: T5 SFT (Camille's pipeline) */}
      <section className="px-4 md:px-12 pb-10 md:pb-12">
        <div className="border-t border-border-light pt-10 md:pt-12 max-w-4xl">
          <span
            className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3"
            style={{ fontFamily: "var(--font-jetbrains-mono)" }}
          >
            Section 01b
          </span>
          <h2
            className="text-[24px] md:text-[32px] leading-[1.05] tracking-[-0.32px] md:tracking-[-0.48px] text-foreground mb-3"
            style={{ fontFamily: "var(--font-dm-serif)" }}
          >
            T5 Supervised Fine-Tuning
          </h2>
          <p className="text-[14px] md:text-[16px] leading-[1.6] text-foreground-secondary mb-5">
            Camille&apos;s T5 pipeline runs in a{" "}
            <a
              href="https://github.com/camille-readbean/CS4248-project-AY2526S2"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent-blue hover:underline"
            >
              separate repo
            </a>{" "}
            with its own stratified split, SLURM orchestration, and a longer
            max-sequence length to fit the combined input+target with the
            strategy token. Same HuggingFace{" "}
            <code className="text-[13px] text-accent-purple">Seq2SeqTrainer</code>{" "}
            as the BART side, but different hyperparameters — the table below
            is the shared recipe; each model card then notes where it differs.
          </p>

          {/* Shared T5 recipe card */}
          <div className="border border-border-card rounded-[22px] p-5 md:p-6 mb-6 md:mb-8">
            <h4
              className="text-[11px] tracking-[0.28px] uppercase text-muted mb-3"
              style={{ fontFamily: "var(--font-jetbrains-mono)" }}
            >
              Shared T5 recipe (joint · control · ablations)
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1.5 text-[12px] md:text-[13px]">
              {[
                ["Data prep", "prepare_t5_datasets.py (80/10/10 stratified)"],
                ["Trainer", "Seq2SeqTrainer, predict_with_generate=True"],
                ["Epochs", "4 (no early stopping)"],
                ["Per-device batch", "8"],
                ["Grad accum", "2 (effective 16)"],
                ["Learning rate", "3e-4"],
                ["Scheduler", "Cosine, warmup_ratio 0.06"],
                ["Weight decay", "0.01"],
                ["Max source/target", "1248 tokens"],
                ["Best metric", "eval_loss"],
                ["Precision", "fp16"],
                ["Seed", "42"],
                ["Compute", "1× NV GPU, 32G mem, SLURM gpu-long, 5h limit"],
                ["Orchestration", "slurm_finetune_t5.sh → .py"],
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
                  <span className="text-foreground-secondary text-right">{v}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-4 md:space-y-5">
            {T5_MODELS.map((m) => (
              <ModelCard key={m.key} m={m} />
            ))}
          </div>
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
            Six retrainings of the T5 control recipe (not joint — the
            ablations use the plain{" "}
            <code className="text-[12px] text-accent-purple">
              rewrite to non-sarcastic:
            </code>{" "}
            prefix and emit the plain rewrite), each with one sarcasm subtype
            dropped from the training data. To keep effective dataset size
            constant across the six variants, every ablation pool is stratified-
            sampled down to the minimum across all drops. The finding on the{" "}
            <Link href="/dashboard" className="text-accent-blue hover:underline">
              dashboard
            </Link>
            : the six ablations cluster within{" "}
            <span className="tabular-nums">0.005</span> of each other on
            similarity — the model learns generic sarcasm patterns that
            transfer across subtypes, so no single one is load-bearing.
          </p>

          <div className="border border-border-card rounded-[22px] p-5 md:p-6 mb-5">
            <h4 className="text-[14px] md:text-[15px] text-foreground mb-3">
              Ablation-specific recipe
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1.5 text-[12px] md:text-[13px]">
              {[
                ["Base", "t5-small (default in slurm script)"],
                ["Input prefix", '"rewrite to non-sarcastic: "'],
                ["Target format", "Plain rewrite (no strategy token)"],
                ["Train pool", "Stratified downsample to min-across-drops"],
                ["Val pool", "Stratified downsample, same rule"],
                ["Test set", "Full held-out split (shared across all six)"],
                ["Everything else", "Same as T5 shared recipe above"],
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
