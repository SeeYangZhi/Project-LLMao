// In static mode (deployed to Vercel), the frontend reads pre-exported
// JSON from /data/* and does pagination/filtering client-side. In dev mode
// it talks to the FastAPI backend. The function signatures stay identical
// so pages don't need to know the difference.

const STATIC_MODE = process.env.NEXT_PUBLIC_USE_STATIC === "true";
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchAPI<T>(path: string): Promise<T> {
  const url = STATIC_MODE ? path : `${API_BASE}${path}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error: ${res.status} ${url}`);
  return res.json();
}

// ─── Metrics ───────────────────────────────────────────────────────────────

export async function getMetricsSummary() {
  if (STATIC_MODE) {
    return fetchAPI<{
      models: Record<string, Record<string, number>>;
      baselines: Record<string, number>;
      registry: Record<string, { display: string; type: string }>;
    }>("/data/metrics-summary.json");
  }
  return fetchAPI<{
    models: Record<string, Record<string, number>>;
    baselines: Record<string, number>;
    registry: Record<string, { display: string; type: string }>;
  }>("/api/metrics/summary");
}

export async function getMetricsByStrategy(model?: string) {
  if (STATIC_MODE) {
    const all = await fetchAPI<
      Record<string, Record<string, Record<string, number>>>
    >("/data/metrics-by-strategy.json");
    if (model) return (all[model] || {}) as unknown as Record<
      string,
      Record<string, Record<string, number>>
    >;
    return all;
  }
  const q = model ? `?model=${model}` : "";
  return fetchAPI<Record<string, Record<string, Record<string, number>>>>(
    `/api/metrics/by-strategy${q}`
  );
}

export async function getModels() {
  if (STATIC_MODE) {
    return fetchAPI<
      { name: string; display: string; type: string; sample_count: number }[]
    >("/data/metrics-models.json");
  }
  return fetchAPI<
    { name: string; display: string; type: string; sample_count: number }[]
  >("/api/metrics/models");
}

// ─── Samples (with client-side filter/sort/paginate in static mode) ────────

const _samplesCache: Record<string, Record<string, unknown>[]> = {};

async function loadSamplesFor(model: string) {
  if (!_samplesCache[model]) {
    _samplesCache[model] = await fetchAPI<Record<string, unknown>[]>(
      `/data/samples/${model}.json`
    );
  }
  return _samplesCache[model];
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
  if (STATIC_MODE) {
    const model = params.model || "bart_base_ce_rl";
    let rows = [...(await loadSamplesFor(model))];
    if (params.strategy) {
      const strategies = params.strategy.split(",");
      rows = rows.filter((r) => strategies.includes(r.subtype as string));
    }
    if (params.search) {
      const q = params.search.toLowerCase();
      rows = rows.filter(
        (r) =>
          ((r.input as string) || "").toLowerCase().includes(q) ||
          ((r.output as string) || "").toLowerCase().includes(q)
      );
    }
    if (params.sort_by) {
      const key = params.sort_by;
      const asc = (params.sort_order || "asc") === "asc";
      rows.sort((a, b) => {
        const av = a[key];
        const bv = b[key];
        if (av === bv) return 0;
        if (av === null || av === undefined) return 1;
        if (bv === null || bv === undefined) return -1;
        return (av < bv ? -1 : 1) * (asc ? 1 : -1);
      });
    }
    const total = rows.length;
    const page = params.page || 1;
    const page_size = params.page_size || 20;
    const start = (page - 1) * page_size;
    return {
      items: rows.slice(start, start + page_size),
      total,
      page,
      page_size,
    };
  }
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
  if (STATIC_MODE) {
    const result: Record<string, Record<string, unknown>> = {};
    // Need to inspect every model file. Use the registry to discover them.
    const models = await getModels();
    await Promise.all(
      models.map(async (m) => {
        const rows = await loadSamplesFor(m.name);
        const row = rows.find((r) => r.id === id);
        if (row) result[m.name] = row;
      })
    );
    return result;
  }
  return fetchAPI<Record<string, Record<string, unknown>>>(
    `/api/samples/${id}/compare`
  );
}

// ─── Inference (only works in dev mode) ────────────────────────────────────

export async function generate(text: string, model: string) {
  if (STATIC_MODE) {
    throw new Error(
      "Live inference is disabled in the hosted version. Run the FastAPI backend locally to use the playground."
    );
  }
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
  if (STATIC_MODE) {
    return fetchAPI<
      {
        name: string;
        display: string;
        available: boolean;
        loaded: boolean;
        note: string;
      }[]
    >("/data/inference-models.json");
  }
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

// ─── Human eval ────────────────────────────────────────────────────────────

export async function getHumanEvalGold(page = 1, pageSize = 20) {
  if (STATIC_MODE) {
    const rows = await fetchAPI<Record<string, unknown>[]>(
      "/data/human-eval/gold.json"
    );
    const total = rows.length;
    const start = (page - 1) * pageSize;
    return {
      items: rows.slice(start, start + pageSize),
      total,
      page,
      page_size: pageSize,
    };
  }
  return fetchAPI<{
    items: Record<string, unknown>[];
    total: number;
    page: number;
    page_size: number;
  }>(`/api/human-eval/gold?page=${page}&page_size=${pageSize}`);
}

export async function getHumanEvalFlagged(model?: string) {
  if (STATIC_MODE) {
    const all = await fetchAPI<
      Record<string, { items: Record<string, unknown>[]; total: number }>
    >("/data/human-eval/flagged.json");
    if (model) return { [model]: all[model] || { items: [], total: 0 } };
    return all;
  }
  const q = model ? `?model=${model}` : "";
  return fetchAPI<
    Record<string, { items: Record<string, unknown>[]; total: number }>
  >(`/api/human-eval/flagged${q}`);
}

export async function getHumanEvalSummary() {
  if (STATIC_MODE) {
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
    >("/data/human-eval/summary.json");
  }
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

export async function getHeldout(params?: {
  strategy?: string;
  page?: number;
  page_size?: number;
}) {
  if (STATIC_MODE) {
    let rows = await fetchAPI<Record<string, unknown>[]>(
      "/data/human-eval/heldout.json"
    );
    if (params?.strategy) {
      rows = rows.filter((r) => r.strategy === params.strategy);
    }
    const total = rows.length;
    const page = params?.page || 1;
    const page_size = params?.page_size || 20;
    const start = (page - 1) * page_size;
    return {
      items: rows.slice(start, start + page_size),
      total,
      page,
      page_size,
    };
  }
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

// ─── Mislabels ─────────────────────────────────────────────────────────────

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
  if (STATIC_MODE) {
    return fetchAPI<{
      total: number;
      over_labeled: number;
      under_labeled: number;
      theonion: number;
      huffpost: number;
      both_models_high_confidence: number;
    }>("/data/mislabels-summary.json");
  }
  return fetchAPI<{
    total: number;
    over_labeled: number;
    under_labeled: number;
    theonion: number;
    huffpost: number;
    both_models_high_confidence: number;
  }>("/api/mislabels/summary");
}

let _mislabelsCache: Mislabel[] | null = null;

export async function getMislabels(params: {
  direction?: string;
  source?: string;
  confidence?: string;
  search?: string;
  page?: number;
  page_size?: number;
}) {
  if (STATIC_MODE) {
    if (!_mislabelsCache) {
      _mislabelsCache = await fetchAPI<Mislabel[]>("/data/mislabels.json");
    }
    let rows = _mislabelsCache;
    if (params.direction === "over" || params.direction === "under") {
      rows = rows.filter((r) => r.direction === params.direction);
    }
    if (params.source === "theonion" || params.source === "huffpost") {
      rows = rows.filter((r) => r.source === params.source);
    }
    if (params.confidence === "high") {
      rows = rows.filter(
        (r) =>
          r.stepfun_confidence === "high" && r.nemotron_confidence === "high"
      );
    }
    if (params.search) {
      const q = params.search.toLowerCase();
      rows = rows.filter((r) => (r.headline || "").toLowerCase().includes(q));
    }
    const total = rows.length;
    const page = params.page || 1;
    const page_size = params.page_size || 15;
    const start = (page - 1) * page_size;
    return {
      items: rows.slice(start, start + page_size),
      total,
      page,
      page_size,
    };
  }
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
