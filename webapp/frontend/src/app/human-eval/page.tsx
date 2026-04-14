"use client";

import { useEffect, useState } from "react";
import {
  getGoldenEval,
  getMultiClassifier,
  getHeldout,
  type GoldenSummary,
  type GoldenClassifierBreakdown,
  type GoldenSubtypeRow,
  type ClassifierFlip,
} from "@/lib/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";

type Tab = "golden" | "classifiers" | "heldout";

const CLASSIFIER_COLORS: Record<string, string> = {
  "RoBERTa-Twitter": "#1863dc",
  "Bert-Kaggle": "#9b60aa",
  "RoBERTa-News": "#0f3460",
};

const MODEL_DISPLAY: Record<string, string> = {
  t5_base_joint: "T5-Joint",
  t5_base_control: "T5-Control",
  bart_base_rl: "BART-RL",
};

export default function HumanEvalPage() {
  const [tab, setTab] = useState<Tab>("golden");
  const [golden, setGolden] = useState<{
    summary: GoldenSummary[];
    classifier_breakdown: GoldenClassifierBreakdown[];
    subtype: Record<string, GoldenSubtypeRow[]>;
    samples: Record<string, Record<string, unknown>[]>;
  } | null>(null);
  const [multiClf, setMultiClf] = useState<{
    models: Record<string, ClassifierFlip[]>;
    classifiers: string[];
  } | null>(null);
  const [heldoutData, setHeldoutData] = useState<{
    items: Record<string, unknown>[];
    total: number;
  } | null>(null);
  const [heldoutPage, setHeldoutPage] = useState(1);
  const [activeGoldenModel, setActiveGoldenModel] = useState("t5_base_joint");

  useEffect(() => {
    getGoldenEval().then(setGolden).catch(() => {});
    getMultiClassifier().then(setMultiClf).catch(() => {});
    getHeldout({ page: 1, page_size: 15 }).then(setHeldoutData).catch(() => {});
  }, []);

  useEffect(() => {
    getHeldout({ page: heldoutPage, page_size: 15 }).then(setHeldoutData).catch(() => {});
  }, [heldoutPage]);

  return (
    <div className="min-h-screen pb-16 md:pb-20">
      {/* Header */}
      <div className="px-4 md:px-12 pt-8 md:pt-12 pb-6 md:pb-8">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3 md:mb-4"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Annotations &middot; Multi-classifier audit
        </span>
        <h1
          className="text-[36px] md:text-[48px] leading-[1.0] tracking-[-0.72px] md:tracking-[-0.96px] text-foreground mb-4"
          style={{ fontFamily: "var(--font-dm-serif)" }}
        >
          Human Evaluation
        </h1>
        <p className="text-[15px] md:text-[17px] leading-[1.55] text-foreground-secondary max-w-3xl">
          We hand-labeled 140 samples across 3 models with 2 independent
          annotators (κ &gt; 0.8) and ran every output through 3 sarcasm
          classifiers. The classifiers all disagree — and they all disagree
          with humans. This page is the receipts.
        </p>
      </div>

      {/* Headline takeaway band */}
      {golden && (
        <div className="px-4 md:px-12 pb-6 md:pb-8">
          <div className="border border-accent-purple/30 bg-accent-purple/[0.03] rounded-[22px] p-5 md:p-8">
            <span
              className="text-[10px] tracking-[0.28px] uppercase text-accent-purple block mb-2"
              style={{ fontFamily: "var(--font-jetbrains-mono)" }}
            >
              Key finding
            </span>
            <h3 className="text-[20px] md:text-[24px] tracking-[-0.24px] text-foreground mb-3">
              Automated flip rate is not a valid primary metric.
            </h3>
            <p className="text-[14px] md:text-[15px] leading-[1.6] text-foreground-secondary">
              Across 3 classifiers and 3 models we see Cohen&apos;s κ ranging
              from <span className="tabular-nums">−0.11</span> to{" "}
              <span className="tabular-nums">+0.18</span> against human labels.
              Human annotators agree at κ &gt; 0.8. Classifiers detect sarcasm{" "}
              <em>presence</em>, not <em>removal</em> — they judge each output
              in isolation and miss the transformation.
            </p>
          </div>
        </div>
      )}

      {/* Golden summary cards */}
      {golden && (
        <div className="px-4 md:px-12 pb-6 md:pb-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-5">
            {golden.summary.map((s) => {
              const isBest = s.model === "t5_base_joint";
              return (
                <div
                  key={s.model}
                  className={`border rounded-[22px] p-5 md:p-6 ${
                    isBest
                      ? "border-accent-blue/40 bg-accent-blue/[0.03]"
                      : "border-border-card"
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-[16px] md:text-[18px] text-foreground">
                      {s.display}
                    </h4>
                    {isBest && (
                      <span
                        className="text-[10px] tracking-[0.16px] uppercase text-accent-blue px-2 py-0.5 rounded-full bg-accent-blue/10"
                        style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                      >
                        Best overall
                      </span>
                    )}
                  </div>
                  <dl className="space-y-2 text-[13px]">
                    <Row
                      label="Human flip rate"
                      value={`${(s.human_flip_rate * 100).toFixed(1)}%`}
                    />
                    <Row
                      label="Meaning change"
                      value={`${(s.meaning_change_rate * 100).toFixed(1)}%`}
                      tone={
                        s.meaning_change_rate < 0.2
                          ? "good"
                          : s.meaning_change_rate < 0.3
                          ? "ok"
                          : "bad"
                      }
                    />
                    <Row
                      label="Strict success"
                      value={`${(s.strict_success_rate * 100).toFixed(1)}%`}
                      tone={
                        s.strict_success_rate > 0.42
                          ? "good"
                          : s.strict_success_rate > 0.37
                          ? "ok"
                          : "bad"
                      }
                    />
                    <Row
                      label="Annotator κ"
                      value={s.inter_annotator_kappa?.toFixed(3) ?? "—"}
                    />
                    <Row
                      label="Similarity"
                      value={s.mean_similarity.toFixed(3)}
                    />
                  </dl>
                </div>
              );
            })}
          </div>
          <p className="text-[12px] text-muted mt-3">
            <strong>Strict success</strong> = sarcasm removed AND meaning
            preserved. T5-Joint wins because its strategy prefix forces task
            decomposition before generation.
          </p>
        </div>
      )}

      {/* Tab selector */}
      <div className="px-4 md:px-12 pb-6">
        <div className="flex gap-2 flex-wrap">
          {(
            [
              { key: "golden", label: "Golden Eval" },
              { key: "classifiers", label: "Multi-Classifier" },
              { key: "heldout", label: "Heldout Set" },
            ] as { key: Tab; label: string }[]
          ).map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`px-4 py-2 rounded-xl text-[13px] transition-all ${
                tab === t.key
                  ? "bg-foreground text-white"
                  : "text-muted hover:text-accent-blue"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── GOLDEN tab ───────────────────────────────────────────────── */}
      {tab === "golden" && golden && (
        <div className="px-4 md:px-12 space-y-6">
          {/* Model picker */}
          <div className="flex gap-2 flex-wrap">
            {Object.keys(golden.subtype).map((m) => (
              <button
                key={m}
                onClick={() => setActiveGoldenModel(m)}
                className={`px-3 py-1.5 rounded-full text-[12px] border transition-all ${
                  activeGoldenModel === m
                    ? "border-accent-blue text-accent-blue bg-accent-blue/5"
                    : "border-border-card text-muted hover:text-foreground-secondary"
                }`}
              >
                {MODEL_DISPLAY[m] || m}
              </button>
            ))}
          </div>

          {/* Subtype breakdown table */}
          <div className="border border-border-card rounded-[22px] p-5 md:p-6">
            <div className="mb-4">
              <h3 className="text-[18px] md:text-[20px] tracking-[-0.2px] text-foreground">
                Per-subtype breakdown — {MODEL_DISPLAY[activeGoldenModel]}
              </h3>
              <p className="text-[13px] text-muted mt-1">
                Human-labeled flip rate vs each classifier&apos;s flip rate.
                Wider gap = more classifier blindness.
              </p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-[12px] md:text-[13px] min-w-[640px]">
                <thead>
                  <tr className="bg-surface-snow border-b border-border-light">
                    <th className="text-left py-3 px-3 text-muted font-normal">
                      Subtype
                    </th>
                    <th className="text-right py-3 px-3 text-muted font-normal">
                      N
                    </th>
                    <th className="text-right py-3 px-3 text-muted font-normal">
                      Human flip
                    </th>
                    <th className="text-right py-3 px-3 text-muted font-normal">
                      Twitter
                    </th>
                    <th className="text-right py-3 px-3 text-muted font-normal">
                      Kaggle
                    </th>
                    <th className="text-right py-3 px-3 text-muted font-normal">
                      News
                    </th>
                    <th className="text-right py-3 px-3 text-muted font-normal">
                      Best κ
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {(golden.subtype[activeGoldenModel] || []).map((row) => {
                    const bestKappa = Math.max(
                      row.twitter_kappa ?? -1,
                      row.kaggle_kappa ?? -1,
                      row.news_kappa ?? -1
                    );
                    return (
                      <tr
                        key={row.subtype}
                        className="border-b border-border-card hover:bg-surface-snow"
                      >
                        <td
                          className="py-2.5 px-3 text-foreground-secondary"
                          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                        >
                          {row.subtype.replace(/_/g, " ")}
                        </td>
                        <td className="py-2.5 px-3 text-right tabular-nums text-muted">
                          {row.n}
                        </td>
                        <td className="py-2.5 px-3 text-right tabular-nums text-foreground">
                          {(row.human_flip_rate * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3 text-right tabular-nums text-foreground-secondary">
                          {(row.twitter_flip_rate * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3 text-right tabular-nums text-foreground-secondary">
                          {(row.kaggle_flip_rate * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3 text-right tabular-nums text-foreground-secondary">
                          {(row.news_flip_rate * 100).toFixed(1)}%
                        </td>
                        <td
                          className={`py-2.5 px-3 text-right tabular-nums ${
                            bestKappa < 0.1
                              ? "text-red-500"
                              : bestKappa < 0.2
                              ? "text-amber-600"
                              : "text-green-600"
                          }`}
                        >
                          {bestKappa.toFixed(3)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Classifier × model accuracy heat */}
          <div className="border border-border-card rounded-[22px] p-5 md:p-6">
            <h3 className="text-[18px] md:text-[20px] tracking-[-0.2px] text-foreground mb-1">
              Classifier accuracy vs human labels
            </h3>
            <p className="text-[13px] text-muted mb-4">
              The 9-cell breakdown from the evaluation pipeline. 4 of 9 cells
              show negative κ (classifier anti-correlates with humans).
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-[12px] md:text-[13px] min-w-[700px]">
                <thead>
                  <tr className="bg-surface-snow border-b border-border-light">
                    <th className="text-left py-3 px-3 text-muted font-normal">
                      Model
                    </th>
                    <th className="text-left py-3 px-3 text-muted font-normal">
                      Classifier
                    </th>
                    <th className="text-right py-3 px-3 text-muted font-normal">
                      Clf flip
                    </th>
                    <th className="text-right py-3 px-3 text-muted font-normal">
                      Human flip
                    </th>
                    <th className="text-right py-3 px-3 text-muted font-normal">
                      Accuracy
                    </th>
                    <th className="text-right py-3 px-3 text-muted font-normal">
                      F1
                    </th>
                    <th className="text-right py-3 px-3 text-muted font-normal">
                      κ vs human
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {golden.classifier_breakdown.map((r, i) => (
                    <tr
                      key={i}
                      className="border-b border-border-card hover:bg-surface-snow"
                    >
                      <td className="py-2.5 px-3 text-foreground">
                        {MODEL_DISPLAY[r.model] || r.model}
                      </td>
                      <td
                        className="py-2.5 px-3 text-foreground-secondary"
                        style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                      >
                        <span
                          className="inline-block w-2 h-2 rounded-full mr-2"
                          style={{
                            backgroundColor:
                              CLASSIFIER_COLORS[r.classifier] || "#93939f",
                          }}
                        />
                        {r.classifier}
                      </td>
                      <td className="py-2.5 px-3 text-right tabular-nums text-foreground-secondary">
                        {(r.clf_flip_rate * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-3 text-right tabular-nums text-muted">
                        {(r.human_flip_rate * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-3 text-right tabular-nums text-foreground-secondary">
                        {(r.accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-3 text-right tabular-nums text-foreground-secondary">
                        {r.f1.toFixed(3)}
                      </td>
                      <td
                        className={`py-2.5 px-3 text-right tabular-nums ${
                          r.kappa < 0
                            ? "text-red-500"
                            : r.kappa < 0.1
                            ? "text-amber-600"
                            : "text-green-600"
                        }`}
                      >
                        {r.kappa >= 0 ? "+" : ""}
                        {r.kappa.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Why this happens */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-5">
            <InsightCard
              title="Why classifiers can't see removal"
              body="The classifiers were trained for sarcasm detection — 'is this single text sarcastic?' — but we ask them to verify removal: 'did the rewrite become non-sarcastic relative to the input?'. They never see the input/output pair together, so they can't detect the transformation."
            />
            <InsightCard
              title="Why subtypes have different miss rates"
              body="Satire mimics legitimate news format (80% miss rate). Rhetorical questions encode sarcasm in implication. Irony has no surface markers — the contradiction is contextual. Classifiers only see surface lexical patterns."
            />
          </div>
        </div>
      )}

      {/* ── MULTI-CLASSIFIER tab ─────────────────────────────────────── */}
      {tab === "classifiers" && multiClf && (
        <div className="px-4 md:px-12 space-y-6">
          <div className="border border-border-card rounded-[22px] p-5 md:p-6">
            <h3 className="text-[18px] md:text-[20px] tracking-[-0.2px] text-foreground mb-1">
              Same model, three classifiers, three different stories
            </h3>
            <p className="text-[13px] text-muted mb-4">
              Up to <strong>33 percentage points</strong> of disagreement on
              the same outputs. Pick whichever classifier supports your
              hypothesis.
            </p>
            <div className="h-[420px] md:h-[480px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={Object.entries(multiClf.models).map(([model, rows]) => {
                    const obj: Record<string, number | string> = {
                      model: model
                        .replace(/_/g, " ")
                        .replace("ablation without ", "w/o "),
                    };
                    rows.forEach((r) => {
                      obj[r.classifier] = r.flip_rate * 100;
                    });
                    return obj;
                  })}
                  margin={{ top: 8, right: 8, left: 0, bottom: 90 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f2f2f2" />
                  <XAxis
                    dataKey="model"
                    tick={{ fontSize: 10, fill: "#93939f" }}
                    angle={-45}
                    textAnchor="end"
                    height={100}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "#93939f" }}
                    label={{
                      value: "Flip rate %",
                      angle: -90,
                      position: "insideLeft",
                      style: { fontSize: 11, fill: "#93939f" },
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      borderRadius: 12,
                      border: "1px solid #f2f2f2",
                      fontSize: 13,
                    }}
                    formatter={(v: unknown) => `${Number(v).toFixed(1)}%`}
                  />
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                  {multiClf.classifiers.map((clf) => (
                    <Bar
                      key={clf}
                      dataKey={clf}
                      fill={CLASSIFIER_COLORS[clf] || "#93939f"}
                      radius={[4, 4, 0, 0]}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Spread table */}
          <div className="border border-border-card rounded-[22px] p-5 md:p-6">
            <h3 className="text-[18px] md:text-[20px] tracking-[-0.2px] text-foreground mb-1">
              Spread per model
            </h3>
            <p className="text-[13px] text-muted mb-4">
              Difference between the highest and lowest classifier flip rate.
              12 of 14 models show &gt;30 pp spread.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-[12px] md:text-[13px] min-w-[560px]">
                <thead>
                  <tr className="bg-surface-snow border-b border-border-light">
                    <th className="text-left py-3 px-3 text-muted font-normal">
                      Model
                    </th>
                    {multiClf.classifiers.map((c) => (
                      <th
                        key={c}
                        className="text-right py-3 px-3 text-muted font-normal"
                      >
                        {c.split("-")[1]}
                      </th>
                    ))}
                    <th className="text-right py-3 px-3 text-muted font-normal">
                      Spread
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(multiClf.models)
                    .map(([model, rows]) => {
                      const map: Record<string, number> = {};
                      rows.forEach((r) => {
                        map[r.classifier] = r.flip_rate;
                      });
                      const vals = Object.values(map);
                      const spread = (Math.max(...vals) - Math.min(...vals)) * 100;
                      return { model, map, spread };
                    })
                    .sort((a, b) => b.spread - a.spread)
                    .map(({ model, map, spread }) => (
                      <tr
                        key={model}
                        className="border-b border-border-card hover:bg-surface-snow"
                      >
                        <td className="py-2.5 px-3 text-foreground-secondary">
                          {model
                            .replace(/_/g, " ")
                            .replace("ablation without ", "w/o ")}
                        </td>
                        {multiClf.classifiers.map((c) => (
                          <td
                            key={c}
                            className="py-2.5 px-3 text-right tabular-nums text-foreground-secondary"
                          >
                            {((map[c] ?? 0) * 100).toFixed(1)}%
                          </td>
                        ))}
                        <td
                          className={`py-2.5 px-3 text-right tabular-nums ${
                            spread > 30
                              ? "text-red-500"
                              : spread > 15
                              ? "text-amber-600"
                              : "text-green-600"
                          }`}
                        >
                          {spread.toFixed(1)} pp
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── HELDOUT tab ─────────────────────────────────────────────── */}
      {tab === "heldout" && heldoutData && (
        <div className="px-4 md:px-12">
          <div className="border border-border-card rounded-[22px] overflow-x-auto">
            <table className="w-full text-[13px] min-w-[700px]">
              <thead>
                <tr className="bg-surface-snow border-b border-border-light">
                  <th className="text-left py-3 px-3 text-muted font-normal w-12">
                    ID
                  </th>
                  <th className="text-left py-3 px-3 text-muted font-normal">
                    Input (Sarcastic)
                  </th>
                  <th className="text-left py-3 px-3 text-muted font-normal">
                    Expected Output (Human)
                  </th>
                  <th className="text-left py-3 px-3 text-muted font-normal w-32">
                    Strategy
                  </th>
                </tr>
              </thead>
              <tbody>
                {heldoutData.items.map((item) => (
                  <tr
                    key={item.id as number}
                    className="border-b border-border-card hover:bg-surface-snow transition-colors"
                  >
                    <td className="py-2.5 px-3 text-muted tabular-nums">
                      {item.id as number}
                    </td>
                    <td className="py-2.5 px-3 text-foreground-secondary">
                      {item.input_text as string}
                    </td>
                    <td className="py-2.5 px-3 text-foreground-secondary">
                      {item.expected_output as string}
                    </td>
                    <td className="py-2.5 px-3">
                      <span
                        className="text-[11px] tracking-[0.16px] uppercase text-muted px-2 py-0.5 rounded-full border border-border-card"
                        style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                      >
                        {item.strategy as string}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="flex items-center justify-center gap-2 py-4 border-t border-border-card">
              <button
                onClick={() => setHeldoutPage((p) => Math.max(1, p - 1))}
                disabled={heldoutPage === 1}
                className="px-3 py-1 rounded-xl text-[13px] text-muted hover:text-accent-blue disabled:opacity-30"
              >
                Previous
              </button>
              <span className="text-[13px] text-muted px-4">
                {heldoutPage} / {Math.ceil(heldoutData.total / 15)}
              </span>
              <button
                onClick={() => setHeldoutPage((p) => p + 1)}
                disabled={heldoutPage >= Math.ceil(heldoutData.total / 15)}
                className="px-3 py-1 rounded-xl text-[13px] text-muted hover:text-accent-blue disabled:opacity-30"
              >
                Next
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Row({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "good" | "ok" | "bad";
}) {
  const valueColor =
    tone === "good"
      ? "text-green-600"
      : tone === "bad"
      ? "text-red-500"
      : tone === "ok"
      ? "text-amber-600"
      : "text-foreground";
  return (
    <div className="flex items-baseline justify-between">
      <dt className="text-muted">{label}</dt>
      <dd className={`tabular-nums ${valueColor}`}>{value}</dd>
    </div>
  );
}

function InsightCard({ title, body }: { title: string; body: string }) {
  return (
    <div className="border border-border-card rounded-[22px] p-5 md:p-6">
      <h4 className="text-[15px] md:text-[16px] text-foreground mb-2">
        {title}
      </h4>
      <p className="text-[13px] md:text-[14px] leading-[1.6] text-foreground-secondary">
        {body}
      </p>
    </div>
  );
}
