# Project LLMao Webapp — Frontend

Next.js 16 + Tailwind + Recharts dashboard for the CS4248 sarcasm style
transfer project. Runs in one of two modes depending on a single env var.

## Two modes

### Local (dev) mode

Talks to the FastAPI backend at `http://localhost:8000`. Required for the
Playground (live BART + LLaMA inference).

```bash
# Terminal 1 — backend
cd ../backend
../../.venv/bin/python -m uvicorn app.main:app --port 8000

# Terminal 2 — frontend
npm run dev
```

Open `http://localhost:3000`.

### Static (Vercel) mode

Reads pre-exported JSON from `public/data/*` instead of calling the backend.
Pagination, filtering, and sorting all happen client-side. The Playground
shows a banner explaining that live inference is disabled.

```bash
NEXT_PUBLIC_USE_STATIC=true npm run dev
```

To deploy to Vercel:

1. Connect the repo to a Vercel project.
2. **Settings → General → Root Directory**: set to `webapp/frontend`. This
   is the only manual UI step — Vercel's Root Directory cannot be set from
   `vercel.json`.
3. Push. The committed `vercel.json` provides the `NEXT_PUBLIC_USE_STATIC`
   build-time env var and the Next.js framework hint, so everything else
   is reproducible from the repo.

## Refreshing the static data

The static JSON lives in `public/data/` and is committed to git. Whenever
the underlying CSVs/JSONLs change, regenerate it:

```bash
cd ../..
.venv/bin/python webapp/backend/scripts/export_static.py
```

This dumps fresh JSON for every endpoint into `public/data/`. Commit the
diff and Vercel will redeploy automatically.

Output is roughly 20 MB across 23 files — well within Vercel's static asset
limits. Each per-model sample file is ~1.4 MB.

## Pages

- `/` — landing
- `/pipeline` — six-stage data pipeline visualization, with file links
- `/mislabels` — 4,076 NHDSD audit cases, browsable with filters
- `/dashboard` — 14 models × 7 metrics, charts and aggregate table
- `/explorer` — per-model sample browser with filters and side-by-side
  comparison
- `/playground` — live BART/LLaMA inference (local mode only)
- `/human-eval` — gold standard, flagged samples, and heldout browser

## Stack

- Next.js 16 (App Router) on React 19
- Tailwind v4
- Recharts for charts
- DM Serif Display + DM Sans + JetBrains Mono via `next/font/google`
