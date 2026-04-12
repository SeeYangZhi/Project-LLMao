const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchAPI<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function getMetricsSummary() {
  return fetchAPI<{
    models: Record<string, Record<string, number>>;
    baselines: Record<string, number>;
    registry: Record<string, { display: string; type: string }>;
  }>("/api/metrics/summary");
}

export async function getMetricsByStrategy(model?: string) {
  const q = model ? `?model=${model}` : "";
  return fetchAPI<Record<string, Record<string, Record<string, number>>>>(`/api/metrics/by-strategy${q}`);
}

export async function getModels() {
  return fetchAPI<
    { name: string; display: string; type: string; sample_count: number }[]
  >("/api/metrics/models");
}

export async function getSamples(params: {
  model?: string;
  strategy?: string;
  search?: string;
  sort_by?: string;
  sort_order?: string;
  page?: number;
  page_size?: number;
}) {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined) q.set(k, String(v));
  });
  return fetchAPI<{
    items: Record<string, unknown>[];
    total: number;
    page: number;
    page_size: number;
  }>(`/api/samples?${q}`);
}

export async function compareSample(id: number) {
  return fetchAPI<Record<string, Record<string, unknown>>>(
    `/api/samples/${id}/compare`
  );
}

export async function generate(text: string, model: string) {
  const res = await fetch(`${API_BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, model }),
  });
  if (!res.ok) throw new Error(`Generate error: ${res.status}`);
  return res.json() as Promise<{
    input: string;
    output: string;
    model: string;
    metrics: Record<string, number>;
    inference_time_ms: number;
  }>;
}

export async function getInferenceModels() {
  return fetchAPI<
    {
      name: string;
      display: string;
      available: boolean;
      loaded: boolean;
      note: string;
    }[]
  >("/api/generate/models");
}

export async function getHumanEvalGold(page = 1, pageSize = 20) {
  return fetchAPI<{
    items: Record<string, unknown>[];
    total: number;
    page: number;
    page_size: number;
  }>(`/api/human-eval/gold?page=${page}&page_size=${pageSize}`);
}

export async function getHumanEvalFlagged(model?: string) {
  const q = model ? `?model=${model}` : "";
  return fetchAPI<
    Record<string, { items: Record<string, unknown>[]; total: number }>
  >(`/api/human-eval/flagged${q}`);
}

export async function getHumanEvalSummary() {
  return fetchAPI<
    Record<
      string,
      {
        total_samples: number;
        flagged_count: number;
        mean_suspicion_score?: number;
        annotator_agreement?: number;
        annotated_count?: number;
      }
    >
  >("/api/human-eval/summary");
}

export type Mislabel = {
  id: number;
  headline: string;
  article_link: string;
  original_label: number;
  stepfun_label: number;
  nemotron_label: number;
  stepfun_confidence: string | null;
  nemotron_confidence: string | null;
  direction: "over" | "under";
  source: "theonion" | "huffpost";
};

export async function getMislabelSummary() {
  return fetchAPI<{
    total: number;
    over_labeled: number;
    under_labeled: number;
    theonion: number;
    huffpost: number;
    both_models_high_confidence: number;
  }>("/api/mislabels/summary");
}

export async function getMislabels(params: {
  direction?: string;
  source?: string;
  confidence?: string;
  search?: string;
  page?: number;
  page_size?: number;
}) {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== "") q.set(k, String(v));
  });
  return fetchAPI<{
    items: Mislabel[];
    total: number;
    page: number;
    page_size: number;
  }>(`/api/mislabels?${q}`);
}

export async function getHeldout(params?: {
  strategy?: string;
  page?: number;
  page_size?: number;
}) {
  const q = new URLSearchParams();
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined) q.set(k, String(v));
    });
  }
  return fetchAPI<{
    items: Record<string, unknown>[];
    total: number;
    page: number;
    page_size: number;
  }>(`/api/human-eval/heldout?${q}`);
}
