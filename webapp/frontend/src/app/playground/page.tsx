"use client";

import { useEffect, useState } from "react";
import { generate, getInferenceModels } from "@/lib/api";

type InferenceModel = {
  name: string;
  display: string;
  available: boolean;
  loaded: boolean;
  note: string;
};

type GenerationResult = {
  input: string;
  output: string;
  model: string;
  metrics: Record<string, number>;
  inference_time_ms: number;
};

export default function PlaygroundPage() {
  const [models, setModels] = useState<InferenceModel[]>([]);
  const [selectedModel, setSelectedModel] = useState("bart-ce-rl");
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<GenerationResult[]>([]);

  useEffect(() => {
    getInferenceModels().then(setModels).catch(() => {});
  }, []);

  const handleGenerate = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await generate(inputText.trim(), selectedModel);
      setResult(res);
      setHistory((prev) => [res, ...prev].slice(0, 10));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed");
    }
    setLoading(false);
  };

  const EXAMPLE_HEADLINES = [
    "Area Man Passionate Defender Of What He Imagines Constitution To Be",
    "Nation's Dog Owners Resolve To Be More Forgiving After Learning How Hard It Is To Be A Dog",
    "Study Finds Every Style Of Parenting Produces Disturbed, Miserable Adults",
    "Report: Most College Males Admit To Regularly Getting Wasted On Knowledge",
  ];

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <div className="px-12 pt-12 pb-8">
        <span
          className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-4"
          style={{ fontFamily: "var(--font-jetbrains-mono)" }}
        >
          Interactive
        </span>
        <h1
          className="text-[48px] leading-[1.0] tracking-[-0.96px] text-foreground"
          style={{ fontFamily: "var(--font-dm-serif)" }}
        >
          Playground
        </h1>
      </div>

      <div className="px-12 grid grid-cols-[1fr_360px] gap-8">
        {/* Main panel */}
        <div className="space-y-6">
          {/* Model selector + status */}
          <div className="flex items-center gap-4">
            <div className="flex gap-2">
              {models.map((m) => (
                <button
                  key={m.name}
                  onClick={() => m.available && setSelectedModel(m.name)}
                  className={`px-4 py-2 rounded-xl text-[13px] border transition-all flex items-center gap-2 ${
                    selectedModel === m.name
                      ? "border-accent-blue text-accent-blue bg-accent-blue/5"
                      : m.available
                      ? "border-border-card text-foreground-secondary hover:border-border"
                      : "border-border-card text-muted/40 cursor-not-allowed"
                  }`}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      m.available ? "bg-green-500" : "bg-border"
                    }`}
                  />
                  {m.display}
                </button>
              ))}
              {models.length === 0 && (
                <span className="text-[13px] text-muted">
                  Checking model availability...
                </span>
              )}
            </div>
          </div>

          {/* Input */}
          <div className="border border-border-card rounded-[22px] p-6">
            <label
              className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3"
              style={{ fontFamily: "var(--font-jetbrains-mono)" }}
            >
              Sarcastic Headline
            </label>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Type a sarcastic news headline..."
              rows={3}
              className="w-full text-[16px] leading-[1.5] text-foreground bg-transparent resize-none focus:outline-none placeholder:text-muted/40"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleGenerate();
                }
              }}
            />
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-border-card">
              <span className="text-[12px] text-muted">
                Press Enter to generate
              </span>
              <button
                onClick={handleGenerate}
                disabled={loading || !inputText.trim()}
                className="px-6 py-2 rounded-full bg-foreground text-white text-[14px] hover:bg-foreground-secondary transition-colors disabled:opacity-40"
              >
                {loading ? "Generating..." : "Generate"}
              </button>
            </div>
          </div>

          {/* Examples */}
          <div>
            <span className="text-[12px] text-muted block mb-2">
              Try an example:
            </span>
            <div className="flex gap-2 flex-wrap">
              {EXAMPLE_HEADLINES.map((h, i) => (
                <button
                  key={i}
                  onClick={() => setInputText(h)}
                  className="text-[12px] text-muted hover:text-accent-blue px-3 py-1.5 border border-border-card rounded-xl transition-colors truncate max-w-[300px]"
                >
                  {h}
                </button>
              ))}
            </div>
          </div>

          {/* Output */}
          {error && (
            <div className="border border-red-200 bg-red-50 rounded-[22px] p-6">
              <p className="text-[14px] text-red-600">{error}</p>
              <p className="text-[12px] text-red-400 mt-1">
                Make sure the backend is running on port 8000
              </p>
            </div>
          )}

          {result && (
            <div className="border border-border-card rounded-[22px] p-6">
              <span
                className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-3"
                style={{ fontFamily: "var(--font-jetbrains-mono)" }}
              >
                Non-Sarcastic Rewrite
              </span>
              <p className="text-[18px] leading-[1.5] text-foreground mb-6">
                {result.output}
              </p>

              {/* Metrics */}
              <div className="flex gap-4 flex-wrap">
                {Object.entries(result.metrics).map(([key, value]) => (
                  <div
                    key={key}
                    className="px-3 py-2 border border-border-card rounded-xl"
                  >
                    <div
                      className="text-[10px] tracking-[0.16px] uppercase text-muted"
                      style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                    >
                      {key.replace(/_/g, " ")}
                    </div>
                    <div className="text-[15px] tabular-nums text-foreground-secondary mt-0.5">
                      {value.toFixed(4)}
                    </div>
                  </div>
                ))}
                <div className="px-3 py-2 border border-border-card rounded-xl">
                  <div
                    className="text-[10px] tracking-[0.16px] uppercase text-muted"
                    style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                  >
                    Inference Time
                  </div>
                  <div className="text-[15px] tabular-nums text-foreground-secondary mt-0.5">
                    {result.inference_time_ms.toFixed(0)}ms
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* History sidebar */}
        <div>
          <span
            className="text-[11px] tracking-[0.28px] uppercase text-muted block mb-4"
            style={{ fontFamily: "var(--font-jetbrains-mono)" }}
          >
            History
          </span>
          <div className="space-y-3">
            {history.length === 0 ? (
              <p className="text-[13px] text-muted/50">
                No generations yet
              </p>
            ) : (
              history.map((h, i) => (
                <div
                  key={i}
                  className="border border-border-card rounded-xl p-4 cursor-pointer hover:border-border transition-colors"
                  onClick={() => {
                    setInputText(h.input);
                    setResult(h);
                  }}
                >
                  <p className="text-[12px] text-muted truncate mb-1">
                    {h.input}
                  </p>
                  <p className="text-[13px] text-foreground-secondary truncate">
                    {h.output}
                  </p>
                  <div className="flex items-center gap-2 mt-2">
                    <span
                      className="text-[10px] text-muted"
                      style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                    >
                      {h.model}
                    </span>
                    <span className="text-[10px] text-muted">
                      {h.inference_time_ms.toFixed(0)}ms
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
