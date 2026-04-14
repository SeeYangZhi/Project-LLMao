"use client";

import { useEffect, useState, useCallback } from "react";
import { getSamples, compareSample, getModels } from "@/lib/api";
import { STRATEGIES } from "@/lib/constants";

type Sample = Record<string, unknown>;

type ModelOption = { value: string; label: string; type: string };

// Pin the model order so the UI reads main → baseline → ablation, matching
// the dashboard filter order. Fetched dynamically so any model added to the
// backend registry shows up here automatically.
const MODEL_TYPE_ORDER = ["main", "baseline", "ablation"];

export default function ExplorerPage() {
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([]);
  const [model, setModel] = useState("t5_base_joint");
  const [strategy, setStrategy] = useState("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState("id");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const [data, setData] = useState<{
    items: Sample[];
    total: number;
  } | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [compareData, setCompareData] = useState<Record<
    string,
    Record<string, unknown>
  > | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    const result = await getSamples({
      model,
      strategy: strategy || undefined,
      search: search || undefined,
      sort_by: sortBy,
      sort_order: sortOrder,
      page,
      page_size: 15,
    });
    setData(result);
    setLoading(false);
  }, [model, strategy, search, sortBy, sortOrder, page]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    getModels()
      .then((models) => {
        const sorted = [...models].sort((a, b) => {
          const ai = MODEL_TYPE_ORDER.indexOf(a.type);
          const bi = MODEL_TYPE_ORDER.indexOf(b.type);
          if (ai !== bi) return ai - bi;
          return a.display.localeCompare(b.display);
        });
        setModelOptions(
          sorted.map((m) => ({ value: m.name, label: m.display, type: m.type }))
        );
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    setPage(1);
  }, [model, strategy, search, sortBy, sortOrder]);

  const handleExpand = async (id: number) => {
    if (expandedId === id) {
      setExpandedId(null);
      setCompareData(null);
      return;
    }
    setExpandedId(id);
    const result = await compareSample(id);
    setCompareData(result);
  };

  const handleSort = (col: string) => {
    if (sortBy === col) {
      setSortOrder((o) => (o === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(col);
      setSortOrder("desc");
    }
  };

  const totalPages = data ? Math.ceil(data.total / 15) : 0;

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <div className="px-4 md:px-12 pt-8 md:pt-12 pb-6 md:pb-8">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3 md:mb-4"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Browse Outputs
        </span>
        <h1
          className="text-[36px] md:text-[48px] leading-[1.0] tracking-[-0.72px] md:tracking-[-0.96px] text-foreground"
          style={{ fontFamily: "var(--font-dm-serif)" }}
        >
          Explorer
        </h1>
      </div>

      {/* Filters */}
      <div className="px-4 md:px-12 pb-6 flex items-center gap-3 md:gap-4 flex-wrap">
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="px-3 py-2 border border-border-card rounded-xl text-[13px] bg-white focus:outline-none focus:border-accent-blue"
        >
          {MODEL_TYPE_ORDER.map((group) => {
            const groupOptions = modelOptions.filter((o) => o.type === group);
            if (groupOptions.length === 0) return null;
            const groupLabel =
              group === "main"
                ? "BART + LLaMA"
                : group === "baseline"
                ? "T5"
                : "Ablations";
            return (
              <optgroup key={group} label={groupLabel}>
                {groupOptions.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </optgroup>
            );
          })}
        </select>

        <select
          value={strategy}
          onChange={(e) => setStrategy(e.target.value)}
          className="px-3 py-2 border border-border-card rounded-xl text-[13px] bg-white focus:outline-none focus:border-accent-blue"
        >
          <option value="">All Strategies</option>
          {STRATEGIES.map((s) => (
            <option key={s.key} value={s.key}>
              {s.label}
            </option>
          ))}
        </select>

        <input
          type="text"
          placeholder="Search headlines..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="px-4 py-2 border border-border-card rounded-xl text-[13px] w-full sm:w-64 bg-white focus:outline-none focus:border-accent-blue placeholder:text-muted/50"
        />

        {data && (
          <span className="text-[12px] md:text-[13px] text-muted md:ml-auto">
            {data.total.toLocaleString()} samples
          </span>
        )}
      </div>

      {/* Table */}
      <div className="px-4 md:px-12">
        <div className="border border-border-card rounded-[22px] overflow-x-auto">
          <table className="w-full text-[13px] min-w-[800px]">
            <thead>
              <tr className="bg-surface-snow border-b border-border-light">
                {[
                  { key: "id", label: "ID", w: "w-16" },
                  { key: "input", label: "Input", w: "w-auto" },
                  { key: "output", label: "Output", w: "w-auto" },
                  { key: "subtype", label: "Strategy", w: "w-28" },
                  { key: "hard_flipped", label: "Flipped", w: "w-20" },
                  { key: "similarity", label: "Sim", w: "w-20" },
                  { key: "flip_delta", label: "Delta", w: "w-20" },
                ].map((col) => (
                  <th
                    key={col.key}
                    className={`text-left py-3 px-3 text-muted font-normal cursor-pointer hover:text-accent-blue transition-colors ${col.w}`}
                    onClick={() => handleSort(col.key)}
                  >
                    {col.label}
                    {sortBy === col.key && (
                      <span className="ml-1">
                        {sortOrder === "asc" ? "^" : "v"}
                      </span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={7} className="py-12 text-center text-muted">
                    Loading...
                  </td>
                </tr>
              ) : (
                data?.items.map((item) => {
                  const id = item.id as number;
                  const isExpanded = expandedId === id;
                  return (
                    <tr key={id} className="group">
                      <td colSpan={7} className="p-0">
                        <div
                          className={`grid grid-cols-[64px_1fr_1fr_112px_80px_80px_80px] items-center border-b border-border-card cursor-pointer transition-colors ${
                            isExpanded
                              ? "bg-accent-blue/[0.03]"
                              : "hover:bg-surface-snow"
                          }`}
                          onClick={() => handleExpand(id)}
                        >
                          <div className="py-2.5 px-3 text-muted tabular-nums">
                            {id}
                          </div>
                          <div className="py-2.5 px-3 truncate text-foreground-secondary">
                            {item.input as string}
                          </div>
                          <div className="py-2.5 px-3 truncate text-foreground-secondary">
                            {item.output as string}
                          </div>
                          <div className="py-2.5 px-3">
                            <span
                              className="text-[11px] tracking-[0.16px] uppercase text-muted px-2 py-0.5 rounded-full border border-border-card"
                              style={{
                                fontFamily: "var(--font-jetbrains-mono)",
                              }}
                            >
                              {item.subtype as string}
                            </span>
                          </div>
                          <div className="py-2.5 px-3 text-center">
                            {(item.hard_flipped as number) === 1 ? (
                              <span className="text-green-600">Yes</span>
                            ) : (
                              <span className="text-muted">No</span>
                            )}
                          </div>
                          <div className="py-2.5 px-3 text-right tabular-nums text-foreground-secondary">
                            {((item.similarity as number) ?? 0).toFixed(3)}
                          </div>
                          <div className="py-2.5 px-3 text-right tabular-nums text-foreground-secondary">
                            {((item.flip_delta as number) ?? 0).toFixed(3)}
                          </div>
                        </div>

                        {/* Expanded detail */}
                        {isExpanded && (
                          <div className="bg-surface-snow border-b border-border-card p-6">
                            <div className="grid grid-cols-2 gap-6 mb-6">
                              <div>
                                <span
                                  className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-2"
                                  style={{
                                    fontFamily: "var(--font-jetbrains-mono)",
                                  }}
                                >
                                  Full Input
                                </span>
                                <p className="text-[14px] text-foreground-secondary leading-relaxed">
                                  {item.input as string}
                                </p>
                              </div>
                              <div>
                                <span
                                  className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-2"
                                  style={{
                                    fontFamily: "var(--font-jetbrains-mono)",
                                  }}
                                >
                                  Full Output
                                </span>
                                <p className="text-[14px] text-foreground-secondary leading-relaxed">
                                  {item.output as string}
                                </p>
                              </div>
                            </div>

                            {/* Metrics */}
                            <div className="flex gap-4 mb-6 flex-wrap">
                              {[
                                "flip_delta",
                                "similarity",
                                "bleu",
                                "perplexity",
                                "edit_dist_norm",
                                "paraphrase_score",
                              ].map((m) => (
                                <div
                                  key={m}
                                  className="px-3 py-2 border border-border-card rounded-xl bg-white"
                                >
                                  <div className="text-[11px] text-muted uppercase tracking-wide">
                                    {m.replace(/_/g, " ")}
                                  </div>
                                  <div className="text-[15px] tabular-nums text-foreground-secondary mt-0.5">
                                    {typeof item[m] === "number"
                                      ? (item[m] as number).toFixed(4)
                                      : "-"}
                                  </div>
                                </div>
                              ))}
                            </div>

                            {/* Model comparison */}
                            {compareData && (
                              <div>
                                <span
                                  className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3"
                                  style={{
                                    fontFamily: "var(--font-jetbrains-mono)",
                                  }}
                                >
                                  All Model Outputs
                                </span>
                                <div className="grid grid-cols-2 gap-3">
                                  {Object.entries(compareData).map(
                                    ([mName, mData]) => (
                                      <div
                                        key={mName}
                                        className={`p-3 rounded-xl border ${
                                          mName === model
                                            ? "border-accent-blue/30 bg-accent-blue/[0.02]"
                                            : "border-border-card bg-white"
                                        }`}
                                      >
                                        <div className="flex items-center justify-between mb-1">
                                          <span className="text-[12px] font-medium text-foreground-secondary">
                                            {mName.replace(/_/g, " ")}
                                          </span>
                                          {(mData.hard_flipped as number) ===
                                          1 ? (
                                            <span className="text-[10px] text-green-600 bg-green-50 px-1.5 py-0.5 rounded-full">
                                              flipped
                                            </span>
                                          ) : (
                                            <span className="text-[10px] text-muted bg-surface-snow px-1.5 py-0.5 rounded-full">
                                              not flipped
                                            </span>
                                          )}
                                        </div>
                                        <p className="text-[13px] text-muted leading-relaxed">
                                          {mData.output as string}
                                        </p>
                                      </div>
                                    )
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-2 mt-6">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-3 py-1.5 rounded-xl text-[13px] border border-border-card text-muted hover:text-accent-blue hover:border-accent-blue/30 disabled:opacity-30 transition-all"
            >
              Previous
            </button>
            <span className="text-[13px] text-muted px-4">
              {page} / {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="px-3 py-1.5 rounded-xl text-[13px] border border-border-card text-muted hover:text-accent-blue hover:border-accent-blue/30 disabled:opacity-30 transition-all"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
