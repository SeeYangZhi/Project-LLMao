"use client";

import { useEffect, useState, useCallback } from "react";
import { getMislabels, getMislabelSummary, type Mislabel } from "@/lib/api";

type Summary = {
  total: number;
  over_labeled: number;
  under_labeled: number;
  theonion: number;
  huffpost: number;
  both_models_high_confidence: number;
};

const LABEL_NAME: Record<number, string> = { 1: "sarcastic", 0: "non-sarcastic" };

export default function MislabelsPage() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [items, setItems] = useState<Mislabel[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [direction, setDirection] = useState<"" | "over" | "under">("");
  const [source, setSource] = useState<"" | "theonion" | "huffpost">("");
  const [confidence, setConfidence] = useState<"" | "high">("");
  const [search, setSearch] = useState("");

  useEffect(() => {
    getMislabelSummary().then(setSummary).catch(() => {});
  }, []);

  const fetchData = useCallback(async () => {
    setLoading(true);
    const res = await getMislabels({
      direction: direction || undefined,
      source: source || undefined,
      confidence: confidence || undefined,
      search: search || undefined,
      page,
      page_size: 15,
    });
    setItems(res.items);
    setTotal(res.total);
    setLoading(false);
  }, [direction, source, confidence, search, page]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    setPage(1);
  }, [direction, source, confidence, search]);

  const totalPages = Math.max(1, Math.ceil(total / 15));

  return (
    <div className="min-h-screen pb-16 md:pb-20">
      {/* Header */}
      <section className="px-4 md:px-12 pt-8 md:pt-12 pb-6 md:pb-8">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3 md:mb-4"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Data Quality Audit
        </span>
        <h1
          className="text-[36px] md:text-[48px] leading-[1.0] tracking-[-0.72px] md:tracking-[-0.96px] text-foreground mb-4"
          style={{ fontFamily: "var(--font-dm-serif)" }}
        >
          Suspected Mislabels
        </h1>
        <p className="text-[16px] md:text-[18px] leading-[1.5] text-foreground-secondary max-w-3xl">
          Headlines where both StepFun 3.5 Flash and Nemotron 3 Nano 30B
          disagreed with the original NHDSD label. Each row links back to the
          source article so you can judge for yourself.
        </p>
      </section>

      {/* Summary stats */}
      {summary && (
        <section className="px-4 md:px-12 pb-8 md:pb-10">
          <div className="border border-border-card rounded-[22px] p-5 md:p-8 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-5 md:gap-6">
            {[
              { v: summary.total.toLocaleString(), l: "Total mislabels", s: "Both LLMs disagree with NHDSD" },
              { v: summary.over_labeled.toLocaleString(), l: "False sarcastic", s: "NHDSD: sarcastic · LLMs: non-sarcastic" },
              { v: summary.under_labeled.toLocaleString(), l: "False non-sarcastic", s: "NHDSD: non-sarcastic · LLMs: sarcastic" },
              { v: summary.theonion.toLocaleString(), l: "TheOnion", s: "vs HuffPost: " + summary.huffpost.toLocaleString() },
              { v: summary.both_models_high_confidence.toLocaleString(), l: "Both high-conf", s: "StepFun + Nemotron both high" },
            ].map((stat) => (
              <div key={stat.l}>
                <div
                  className="text-[32px] leading-[1.0] tracking-[-0.64px] text-foreground mb-1.5 tabular-nums"
                  style={{ fontFamily: "var(--font-dm-serif)" }}
                >
                  {stat.v}
                </div>
                <div className="text-[14px] text-foreground-secondary">
                  {stat.l}
                </div>
                <div className="text-[12px] text-muted mt-0.5">{stat.s}</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Filters */}
      <section className="px-4 md:px-12 pb-6 flex items-center gap-3 flex-wrap">
        <div className="flex gap-1.5 flex-wrap">
          {([
            { key: "", label: "All NHDSD labels" },
            { key: "over", label: "NHDSD: sarcastic" },
            { key: "under", label: "NHDSD: non-sarcastic" },
          ] as const).map((opt) => (
            <button
              key={opt.key}
              onClick={() => setDirection(opt.key)}
              className={`px-3 py-1.5 rounded-full text-[12px] border transition-all ${
                direction === opt.key
                  ? "border-accent-blue text-accent-blue bg-accent-blue/5"
                  : "border-border-card text-muted hover:text-foreground-secondary"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <div className="hidden md:block h-4 w-px bg-border-light" />
        <div className="flex gap-1.5 flex-wrap">
          {([
            { key: "", label: "Both sources" },
            { key: "theonion", label: "TheOnion" },
            { key: "huffpost", label: "HuffPost" },
          ] as const).map((opt) => (
            <button
              key={opt.key}
              onClick={() => setSource(opt.key)}
              className={`px-3 py-1.5 rounded-full text-[12px] border transition-all ${
                source === opt.key
                  ? "border-accent-blue text-accent-blue bg-accent-blue/5"
                  : "border-border-card text-muted hover:text-foreground-secondary"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <div className="hidden md:block h-4 w-px bg-border-light" />
        <button
          onClick={() => setConfidence(confidence === "high" ? "" : "high")}
          className={`px-3 py-1.5 rounded-full text-[12px] border transition-all ${
            confidence === "high"
              ? "border-accent-blue text-accent-blue bg-accent-blue/5"
              : "border-border-card text-muted hover:text-foreground-secondary"
          }`}
        >
          Both LLMs high-confidence only
        </button>
        <input
          type="text"
          placeholder="Search headlines..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="px-4 py-1.5 border border-border-card rounded-full text-[12px] w-full sm:w-56 bg-white focus:outline-none focus:border-accent-blue placeholder:text-muted/50 md:ml-auto"
        />
      </section>

      {/* Results */}
      <section className="px-4 md:px-12">
        <div className="text-[13px] text-muted mb-3">
          {loading ? "Loading..." : `${total.toLocaleString()} mislabels`}
        </div>
        <div className="space-y-3">
          {items.map((item) => (
            <div
              key={item.id}
              className="border border-border-card rounded-[22px] p-4 md:p-5 hover:border-border transition-colors"
            >
              <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-3 md:gap-6">
                {/* Left: headline + labels */}
                <div className="flex-1 min-w-0">
                  <p className="text-[15px] md:text-[16px] leading-[1.5] text-foreground mb-3">
                    {item.headline}
                  </p>
                  <div className="flex items-center gap-2 md:gap-3 flex-wrap">
                    <span
                      className="text-[10px] tracking-[0.16px] uppercase text-muted"
                      style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                    >
                      Labels
                    </span>
                    <LabelBadge
                      name="NHDSD"
                      label={LABEL_NAME[item.original_label]}
                      tone="orig"
                    />
                    <span className="text-muted text-[11px]">vs</span>
                    <LabelBadge
                      name="StepFun"
                      label={LABEL_NAME[item.stepfun_label]}
                      tone="llm"
                      confidence={item.stepfun_confidence || undefined}
                    />
                    <LabelBadge
                      name="Nemotron"
                      label={LABEL_NAME[item.nemotron_label]}
                      tone="llm"
                      confidence={item.nemotron_confidence || undefined}
                    />
                  </div>
                </div>

                {/* Right: source + link */}
                <div className="shrink-0 flex flex-row md:flex-col items-center md:items-end gap-3 md:gap-2">
                  <span
                    className="text-[10px] tracking-[0.16px] uppercase px-2 py-0.5 rounded-full border border-border-card text-muted"
                    style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                  >
                    {item.source === "theonion" ? "TheOnion" : "HuffPost"}
                  </span>
                  <a
                    href={item.article_link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[12px] text-accent-blue hover:underline"
                  >
                    read article ↗
                  </a>
                </div>
              </div>
            </div>
          ))}
          {!loading && items.length === 0 && (
            <div className="text-center py-16 text-muted text-[14px]">
              No mislabels match these filters.
            </div>
          )}
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
            <span className="text-[13px] text-muted px-4 tabular-nums">
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
      </section>
    </div>
  );
}

function LabelBadge({
  name,
  label,
  tone,
  confidence,
}: {
  name: string;
  label: string;
  tone: "orig" | "llm";
  confidence?: string;
}) {
  const isSarcastic = label === "sarcastic";
  const baseColor =
    tone === "orig"
      ? "text-foreground-secondary border-border"
      : isSarcastic
      ? "text-accent-purple border-accent-purple/30 bg-accent-purple/5"
      : "text-accent-blue border-accent-blue/30 bg-accent-blue/5";
  return (
    <div
      className={`text-[11px] px-2 py-0.5 rounded-full border ${baseColor} flex items-center gap-1.5`}
      style={{ fontFamily: "var(--font-jetbrains-mono)" }}
    >
      <span className="opacity-60">{name}:</span>
      <span>{label}</span>
      {confidence && (
        <span className="opacity-40">· {confidence}</span>
      )}
    </div>
  );
}
