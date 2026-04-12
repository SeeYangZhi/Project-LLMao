"use client";

import { useEffect, useState } from "react";
import {
  getHumanEvalGold,
  getHumanEvalFlagged,
  getHumanEvalSummary,
  getHeldout,
} from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

type Tab = "gold" | "flagged" | "heldout";

export default function HumanEvalPage() {
  const [tab, setTab] = useState<Tab>("gold");
  const [goldData, setGoldData] = useState<{
    items: Record<string, unknown>[];
    total: number;
  } | null>(null);
  const [flaggedData, setFlaggedData] = useState<Record<
    string,
    { items: Record<string, unknown>[]; total: number }
  > | null>(null);
  const [summary, setSummary] = useState<Record<
    string,
    {
      total_samples: number;
      flagged_count: number;
      mean_suspicion_score?: number;
      annotator_agreement?: number;
      annotated_count?: number;
    }
  > | null>(null);
  const [heldoutData, setHeldoutData] = useState<{
    items: Record<string, unknown>[];
    total: number;
  } | null>(null);
  const [goldPage, setGoldPage] = useState(1);
  const [heldoutPage, setHeldoutPage] = useState(1);

  useEffect(() => {
    getHumanEvalSummary().then(setSummary);
    getHumanEvalGold(1, 15).then(setGoldData);
    getHumanEvalFlagged().then(setFlaggedData);
    getHeldout({ page: 1, page_size: 15 }).then(setHeldoutData);
  }, []);

  useEffect(() => {
    getHumanEvalGold(goldPage, 15).then(setGoldData);
  }, [goldPage]);

  useEffect(() => {
    getHeldout({ page: heldoutPage, page_size: 15 }).then(setHeldoutData);
  }, [heldoutPage]);

  const summaryChartData = summary
    ? Object.entries(summary).map(([model, s]) => ({
        model: model.replace(/_/g, " ").replace("ablation without ", "w/o "),
        flagged: s.flagged_count,
        total: s.total_samples,
        suspicion: s.mean_suspicion_score ?? 0,
      }))
    : [];

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <div className="px-12 pt-12 pb-8">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-4"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Annotations
        </span>
        <h1
          className="text-[48px] leading-[1.0] tracking-[-0.96px] text-foreground"
          style={{ fontFamily: "var(--font-dm-serif)" }}
        >
          Human Evaluation
        </h1>
      </div>

      {/* Summary stats */}
      {summary && (
        <div className="px-12 pb-8">
          <div className="border border-border-card rounded-[22px] p-8">
            <h3 className="text-[20px] tracking-[-0.2px] text-foreground mb-6">
              Flagged Samples per Model
            </h3>
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={summaryChartData}
                  margin={{ top: 8, right: 8, left: 0, bottom: 60 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f2f2f2" />
                  <XAxis
                    dataKey="model"
                    tick={{ fontSize: 10, fill: "#93939f" }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis tick={{ fontSize: 11, fill: "#93939f" }} />
                  <Tooltip
                    contentStyle={{
                      borderRadius: 12,
                      border: "1px solid #f2f2f2",
                      fontSize: 13,
                    }}
                  />
                  <Bar
                    dataKey="flagged"
                    name="Flagged"
                    fill="#ef4444"
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar
                    dataKey="total"
                    name="Total"
                    fill="#e5e7eb"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Tab selector */}
      <div className="px-12 pb-6">
        <div className="flex gap-2">
          {(
            [
              { key: "gold", label: "Gold Standard" },
              { key: "flagged", label: "Flagged Samples" },
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

      {/* Tab content */}
      <div className="px-12">
        {tab === "gold" && goldData && (
          <div className="border border-border-card rounded-[22px] overflow-hidden">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="bg-surface-snow border-b border-border-light">
                  <th className="text-left py-3 px-3 text-muted font-normal w-12">
                    ID
                  </th>
                  <th className="text-left py-3 px-3 text-muted font-normal">
                    Sarcastic Input
                  </th>
                  <th className="text-right py-3 px-3 text-muted font-normal w-24">
                    Sarc Prob
                  </th>
                  <th className="text-left py-3 px-3 text-muted font-normal">
                    BART RL Output
                  </th>
                  <th className="text-right py-3 px-3 text-muted font-normal w-24">
                    RL Prob
                  </th>
                  <th className="text-left py-3 px-3 text-muted font-normal">
                    BART CE+RL Output
                  </th>
                  <th className="text-right py-3 px-3 text-muted font-normal w-24">
                    CE+RL Prob
                  </th>
                </tr>
              </thead>
              <tbody>
                {goldData.items.map((item) => (
                  <tr
                    key={item.id as number}
                    className="border-b border-border-card hover:bg-surface-snow transition-colors"
                  >
                    <td className="py-2.5 px-3 text-muted tabular-nums">
                      {item.id as number}
                    </td>
                    <td className="py-2.5 px-3 text-foreground-secondary max-w-[200px] truncate">
                      {item.sarcastic_input as string}
                    </td>
                    <td className="py-2.5 px-3 text-right tabular-nums text-foreground-secondary">
                      {((item.input_sarc_prob as number) ?? 0).toFixed(3)}
                    </td>
                    <td className="py-2.5 px-3 text-foreground-secondary max-w-[200px] truncate">
                      {item.BART_RL_output as string}
                    </td>
                    <td className="py-2.5 px-3 text-right tabular-nums">
                      <span
                        className={
                          (item.BART_RL_sarc_prob as number) < 0.5
                            ? "text-green-600"
                            : "text-red-500"
                        }
                      >
                        {((item.BART_RL_sarc_prob as number) ?? 0).toFixed(3)}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-foreground-secondary max-w-[200px] truncate">
                      {item.BART_CE_RL_output as string}
                    </td>
                    <td className="py-2.5 px-3 text-right tabular-nums">
                      <span
                        className={
                          (item.BART_CE_RL_sarc_prob as number) < 0.5
                            ? "text-green-600"
                            : "text-red-500"
                        }
                      >
                        {((item.BART_CE_RL_sarc_prob as number) ?? 0).toFixed(
                          3
                        )}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="flex items-center justify-center gap-2 py-4 border-t border-border-card">
              <button
                onClick={() => setGoldPage((p) => Math.max(1, p - 1))}
                disabled={goldPage === 1}
                className="px-3 py-1 rounded-xl text-[13px] text-muted hover:text-accent-blue disabled:opacity-30"
              >
                Previous
              </button>
              <span className="text-[13px] text-muted px-4">
                {goldPage} / {Math.ceil(goldData.total / 15)}
              </span>
              <button
                onClick={() => setGoldPage((p) => p + 1)}
                disabled={goldPage >= Math.ceil(goldData.total / 15)}
                className="px-3 py-1 rounded-xl text-[13px] text-muted hover:text-accent-blue disabled:opacity-30"
              >
                Next
              </button>
            </div>
          </div>
        )}

        {tab === "flagged" && flaggedData && (
          <div className="space-y-6">
            {Object.entries(flaggedData).map(([modelName, data]) => (
              <div
                key={modelName}
                className="border border-border-card rounded-[22px] p-6"
              >
                <div className="flex items-baseline justify-between mb-4">
                  <h3 className="text-[16px] text-foreground">
                    {modelName.replace(/_/g, " ")}
                  </h3>
                  <span className="text-[13px] text-muted">
                    {data.total} flagged
                  </span>
                </div>
                <div className="space-y-3">
                  {data.items.slice(0, 5).map((item) => (
                    <div
                      key={item.id as number}
                      className="p-4 bg-surface-snow rounded-xl"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1 min-w-0">
                          <p className="text-[13px] text-muted mb-1 truncate">
                            {item.input as string}
                          </p>
                          <p className="text-[14px] text-foreground-secondary">
                            {item.output as string}
                          </p>
                        </div>
                        <div className="shrink-0 text-right">
                          <div className="text-[12px] tabular-nums text-foreground-secondary">
                            {((item.suspicion_score as number) ?? 0).toFixed(3)}
                          </div>
                          <div className="text-[11px] text-muted mt-0.5">
                            suspicion
                          </div>
                        </div>
                      </div>
                      {item.flag_reason && (
                        <div className="mt-2 flex gap-1.5 flex-wrap">
                          {(item.flag_reason as string)
                            .split("; ")
                            .map((reason, i) => (
                              <span
                                key={i}
                                className="text-[10px] px-2 py-0.5 rounded-full bg-red-50 text-red-500 border border-red-100"
                              >
                                {reason}
                              </span>
                            ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {tab === "heldout" && heldoutData && (
          <div className="border border-border-card rounded-[22px] overflow-hidden">
            <table className="w-full text-[13px]">
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
        )}
      </div>
    </div>
  );
}
