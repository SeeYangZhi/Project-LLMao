"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  CartesianGrid,
} from "recharts";
import { getMetricsSummary, getMetricsByStrategy } from "@/lib/api";
import { MODEL_COLORS, METRIC_INFO, STRATEGIES, formatMetric } from "@/lib/constants";

type SummaryData = {
  models: Record<string, Record<string, number>>;
  baselines: Record<string, number>;
  registry: Record<string, { display: string; type: string }>;
};

type StrategyData = Record<string, Record<string, Record<string, number>>>;

export default function DashboardPage() {
  const [summary, setSummary] = useState<SummaryData | null>(null);
  const [strategyData, setStrategyData] = useState<StrategyData | null>(null);
  const [activeMetric, setActiveMetric] = useState("hard_flip_rate");
  const [selectedModel, setSelectedModel] = useState("bart_base_ce_rl");
  const [radarModels, setRadarModels] = useState<string[]>([
    "bart_base_ce_rl",
    "llama_3_2_1b",
    "bart_base",
  ]);
  const [modelFilter, setModelFilter] = useState<"all" | "main" | "ablation">(
    "all"
  );

  useEffect(() => {
    getMetricsSummary().then(setSummary);
    getMetricsByStrategy().then(setStrategyData);
  }, []);

  if (!summary) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-muted text-[15px]">Loading dashboard...</div>
      </div>
    );
  }

  const filteredModels = Object.entries(summary.registry).filter(
    ([, meta]) => modelFilter === "all" || meta.type === modelFilter
  );

  // Bar chart data for active metric
  const barData = filteredModels
    .map(([name, meta]) => ({
      name: meta.display,
      value: summary.models[name]?.[activeMetric] ?? 0,
      fill: MODEL_COLORS[name] || "#93939f",
    }))
    .sort((a, b) => b.value - a.value);

  // Radar chart data
  const radarMetrics = [
    "hard_flip_rate",
    "flip_delta",
    "similarity",
    "edit_dist_norm",
  ];
  const radarData = radarMetrics.map((metric) => {
    const row: Record<string, unknown> = {
      metric: METRIC_INFO[metric]?.label || metric,
    };
    const values = Object.values(summary.models).map(
      (m) => m[metric] ?? 0
    );
    const maxVal = Math.max(...values, 0.001);
    radarModels.forEach((model) => {
      const val = summary.models[model]?.[metric] ?? 0;
      row[model] = (val / maxVal) * 100;
    });
    return row;
  });

  // Strategy heatmap data
  const strategyChartData =
    strategyData && strategyData[selectedModel]
      ? STRATEGIES.map((s) => {
          const d = strategyData[selectedModel]?.[s.key] || {};
          return {
            strategy: s.label,
            ...d,
          };
        })
      : [];

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <div className="px-12 pt-12 pb-8">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-4"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Model Comparison
        </span>
        <h1
          className="text-[48px] leading-[1.0] tracking-[-0.96px] text-foreground"
          style={{ fontFamily: "var(--font-dm-serif)" }}
        >
          Dashboard
        </h1>
      </div>

      {/* Metric selector + filter */}
      <div className="px-12 pb-8 flex items-center gap-6 flex-wrap">
        <div className="flex gap-2">
          {(["all", "main", "ablation"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setModelFilter(f)}
              className={`px-4 py-1.5 rounded-full text-[13px] transition-all ${
                modelFilter === f
                  ? "bg-foreground text-white"
                  : "text-muted hover:text-accent-blue"
              }`}
            >
              {f === "all" ? "All Models" : f === "main" ? "Main" : "Ablations"}
            </button>
          ))}
        </div>
        <div className="h-4 w-px bg-border-light" />
        <div className="flex gap-1.5 flex-wrap">
          {Object.entries(METRIC_INFO).map(([key, info]) => (
            <button
              key={key}
              onClick={() => setActiveMetric(key)}
              className={`px-3 py-1 rounded-full text-[12px] border transition-all ${
                activeMetric === key
                  ? "border-accent-blue text-accent-blue bg-accent-blue/5"
                  : "border-border-card text-muted hover:border-border hover:text-foreground-secondary"
              }`}
            >
              {info.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main bar chart */}
      <div className="px-12 pb-12">
        <div className="border border-border-card rounded-[22px] p-8">
          <div className="flex items-baseline justify-between mb-6">
            <h3 className="text-[20px] tracking-[-0.2px] text-foreground">
              {METRIC_INFO[activeMetric]?.label}
            </h3>
            <span className="text-[13px] text-muted">
              {METRIC_INFO[activeMetric]?.description}
            </span>
          </div>
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={barData}
                margin={{ top: 8, right: 8, left: 0, bottom: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f2f2f2" />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 11, fill: "#93939f" }}
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
                  formatter={(value: unknown) => [
                    formatMetric(
                      Number(value),
                      METRIC_INFO[activeMetric]?.format || "decimal"
                    ),
                    METRIC_INFO[activeMetric]?.label,
                  ]}
                />
                <Bar
                  dataKey="value"
                  radius={[6, 6, 0, 0]}
                  fill="#1863dc"
                  isAnimationActive={true}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          {/* Baseline reference */}
          {summary.baselines[activeMetric] !== undefined && (
            <div className="mt-4 flex items-center gap-2 text-[13px] text-muted">
              <span className="w-4 h-px bg-accent-purple inline-block" />
              Gold human baseline:{" "}
              {formatMetric(
                summary.baselines[activeMetric],
                METRIC_INFO[activeMetric]?.format || "decimal"
              )}
            </div>
          )}
        </div>
      </div>

      {/* Two-column: Radar + Strategy */}
      <div className="px-12 grid grid-cols-2 gap-6 pb-12">
        {/* Radar */}
        <div className="border border-border-card rounded-[22px] p-8">
          <h3 className="text-[20px] tracking-[-0.2px] text-foreground mb-2">
            Model Profiles
          </h3>
          <p className="text-[13px] text-muted mb-6">
            Normalized comparison across key metrics
          </p>
          <div className="flex gap-2 mb-4 flex-wrap">
            {Object.entries(summary.registry)
              .filter(([, m]) => m.type === "main")
              .map(([name, meta]) => (
                <button
                  key={name}
                  onClick={() =>
                    setRadarModels((prev) =>
                      prev.includes(name)
                        ? prev.filter((m) => m !== name)
                        : [...prev, name]
                    )
                  }
                  className={`px-2.5 py-1 rounded-full text-[11px] border transition-all ${
                    radarModels.includes(name)
                      ? "border-current text-accent-blue bg-accent-blue/5"
                      : "border-border-card text-muted"
                  }`}
                >
                  {meta.display}
                </button>
              ))}
          </div>
          <div className="h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="#e5e7eb" />
                <PolarAngleAxis
                  dataKey="metric"
                  tick={{ fontSize: 12, fill: "#93939f" }}
                />
                <PolarRadiusAxis tick={false} axisLine={false} />
                {radarModels.map((model) => (
                  <Radar
                    key={model}
                    name={
                      summary.registry[model]?.display || model
                    }
                    dataKey={model}
                    stroke={MODEL_COLORS[model] || "#93939f"}
                    fill={MODEL_COLORS[model] || "#93939f"}
                    fillOpacity={0.1}
                    strokeWidth={2}
                  />
                ))}
                <Legend
                  wrapperStyle={{ fontSize: 12, color: "#93939f" }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Strategy breakdown */}
        <div className="border border-border-card rounded-[22px] p-8">
          <div className="flex items-baseline justify-between mb-2">
            <h3 className="text-[20px] tracking-[-0.2px] text-foreground">
              Strategy Breakdown
            </h3>
            <span className="text-[12px] text-muted">
              {METRIC_INFO[activeMetric]?.label}
            </span>
          </div>
          <p className="text-[13px] text-muted mb-4">
            {METRIC_INFO[activeMetric]?.label} by sarcasm subtype —{" "}
            {METRIC_INFO[activeMetric]?.higher_better
              ? "higher is better"
              : "lower is better"}
          </p>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="mb-6 px-3 py-1.5 border border-border-card rounded-xl text-[13px] text-foreground-secondary bg-white focus:outline-none focus:border-accent-blue"
          >
            {Object.entries(summary.registry).map(([name, meta]) => (
              <option key={name} value={name}>
                {meta.display}
              </option>
            ))}
          </select>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={strategyChartData}
                margin={{ top: 8, right: 8, left: 0, bottom: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f2f2f2" />
                <XAxis
                  dataKey="strategy"
                  tick={{ fontSize: 11, fill: "#93939f" }}
                />
                <YAxis tick={{ fontSize: 11, fill: "#93939f" }} />
                <Tooltip
                  contentStyle={{
                    borderRadius: 12,
                    border: "1px solid #f2f2f2",
                    fontSize: 13,
                  }}
                  formatter={(value: unknown) => [
                    formatMetric(
                      Number(value),
                      METRIC_INFO[activeMetric]?.format || "decimal"
                    ),
                    METRIC_INFO[activeMetric]?.label,
                  ]}
                />
                <Bar
                  dataKey={activeMetric}
                  name={METRIC_INFO[activeMetric]?.label}
                  fill={MODEL_COLORS[selectedModel] || "#1863dc"}
                  radius={[4, 4, 0, 0]}
                  isAnimationActive={true}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          {summary.baselines[activeMetric] !== undefined && (
            <div className="mt-3 flex items-center gap-2 text-[12px] text-muted">
              <span className="w-4 h-px bg-accent-purple inline-block" />
              Gold human baseline:{" "}
              {formatMetric(
                summary.baselines[activeMetric],
                METRIC_INFO[activeMetric]?.format || "decimal"
              )}
            </div>
          )}
        </div>
      </div>

      {/* Aggregate table */}
      <div className="px-12 pb-12">
        <div className="border border-border-card rounded-[22px] p-8 overflow-x-auto">
          <h3 className="text-[20px] tracking-[-0.2px] text-foreground mb-6">
            Aggregate Metrics
          </h3>
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-border-light">
                <th className="text-left py-3 px-3 text-muted font-normal">
                  Model
                </th>
                {Object.entries(METRIC_INFO).map(([key, info]) => (
                  <th
                    key={key}
                    className={`text-right py-3 px-3 font-normal cursor-pointer transition-colors ${
                      activeMetric === key
                        ? "text-accent-blue"
                        : "text-muted hover:text-foreground-secondary"
                    }`}
                    onClick={() => setActiveMetric(key)}
                  >
                    {info.label}
                    {info.higher_better ? " +" : " -"}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredModels.map(([name, meta]) => {
                const metrics = summary.models[name] || {};
                return (
                  <tr
                    key={name}
                    className="border-b border-border-card hover:bg-surface-snow transition-colors"
                  >
                    <td className="py-2.5 px-3 flex items-center gap-2">
                      <span
                        className="w-2.5 h-2.5 rounded-full inline-block"
                        style={{
                          backgroundColor:
                            MODEL_COLORS[name] || "#93939f",
                        }}
                      />
                      <span className="text-foreground-secondary">
                        {meta.display}
                      </span>
                      <span
                        className="text-[10px] tracking-[0.16px] uppercase text-muted/60 ml-1"
                        style={{
                          fontFamily: "var(--font-jetbrains-mono)",
                        }}
                      >
                        {meta.type}
                      </span>
                    </td>
                    {Object.entries(METRIC_INFO).map(([key, info]) => {
                      const val = metrics[key];
                      return (
                        <td
                          key={key}
                          className="text-right py-2.5 px-3 tabular-nums text-foreground-secondary"
                        >
                          {val !== undefined
                            ? formatMetric(val, info.format)
                            : "-"}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
