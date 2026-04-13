"""Export the entire DataStore as static JSON files for Vercel deployment.

Run this whenever the underlying CSVs or JSONLs change. The output lives
in `webapp/frontend/public/data/` and is consumed by the frontend when
`NEXT_PUBLIC_USE_STATIC=true` is set.

Usage (from project root):
    ../../.venv/bin/python webapp/backend/scripts/export_static.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Make the backend app importable when run from anywhere.
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import BASELINES, METRIC_COLUMNS, MODEL_REGISTRY  # noqa: E402
from app.services.data_loader import data_store  # noqa: E402

PROJECT_ROOT = BACKEND_DIR.parent.parent
OUT_DIR = PROJECT_ROOT / "webapp" / "frontend" / "public" / "data"
SAMPLES_DIR = OUT_DIR / "samples"
HUMAN_EVAL_DIR = OUT_DIR / "human-eval"


def _clean(records: list[dict]) -> list[dict]:
    """Replace NaN/inf with None so JSON serializes cleanly."""
    cleaned = []
    for rec in records:
        out = {}
        for k, v in rec.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                out[k] = None
            else:
                out[k] = v
        cleaned.append(out)
    return cleaned


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size:,} bytes)")


def export_metrics():
    print("[metrics]")
    _write(
        OUT_DIR / "metrics-summary.json",
        {
            "models": data_store.summary,
            "baselines": BASELINES,
            "registry": MODEL_REGISTRY,
        },
    )
    _write(OUT_DIR / "metrics-by-strategy.json", data_store.strategy_summary)
    _write(
        OUT_DIR / "metrics-models.json",
        [
            {
                "name": name,
                "display": MODEL_REGISTRY.get(name, {}).get("display", name),
                "type": MODEL_REGISTRY.get(name, {}).get("type", "unknown"),
                "sample_count": len(data_store.results[name]),
            }
            for name in data_store.results
        ],
    )


def export_samples():
    print("[samples]")
    for model_name, df in data_store.results.items():
        records = _clean(df.to_dict(orient="records"))
        _write(SAMPLES_DIR / f"{model_name}.json", records)


def export_mislabels():
    print("[mislabels]")
    items = data_store.mislabels
    over = sum(1 for m in items if m["direction"] == "over")
    under = sum(1 for m in items if m["direction"] == "under")
    onion = sum(1 for m in items if m["source"] == "theonion")
    huffpost = sum(1 for m in items if m["source"] == "huffpost")
    high_conf = sum(
        1
        for m in items
        if m.get("stepfun_confidence") == "high"
        and m.get("nemotron_confidence") == "high"
    )
    _write(
        OUT_DIR / "mislabels-summary.json",
        {
            "total": len(items),
            "over_labeled": over,
            "under_labeled": under,
            "theonion": onion,
            "huffpost": huffpost,
            "both_models_high_confidence": high_conf,
        },
    )
    _write(OUT_DIR / "mislabels.json", items)


def export_multi_classifier():
    print("[multi-classifier]")
    _write(
        OUT_DIR / "multi-classifier.json",
        {
            "models": data_store.multi_classifier,
            "classifiers": ["RoBERTa-Twitter", "DistilBERT-Reddit", "RoBERTa-News"],
        },
    )


def export_golden():
    print("[golden]")
    _write(
        OUT_DIR / "golden-eval.json",
        {
            "summary": _clean(data_store.golden_summary),
            "classifier_breakdown": _clean(data_store.golden_classifier_breakdown),
            "subtype": {k: _clean(v) for k, v in data_store.golden_subtype.items()},
            "samples": {k: _clean(v) for k, v in data_store.golden_samples.items()},
        },
    )


def export_human_eval():
    print("[human-eval]")
    if data_store.gold_eval is not None:
        _write(
            HUMAN_EVAL_DIR / "gold.json",
            _clean(data_store.gold_eval.to_dict(orient="records")),
        )
    else:
        _write(HUMAN_EVAL_DIR / "gold.json", [])

    summary = {}
    for model_name, df in data_store.human_eval.items():
        stats = {
            "total_samples": len(df),
            "flagged_count": int(
                len(df[df["flag_reason"].notna() & (df["flag_reason"] != "")])
            ),
        }
        if "suspicion_score" in df.columns:
            mean = float(df["suspicion_score"].mean())
            stats["mean_suspicion_score"] = round(mean, 4) if not math.isnan(mean) else None
        a1, a2 = "annotator_1_sarcasm_removed", "annotator_2_sarcasm_removed"
        if a1 in df.columns and a2 in df.columns:
            both = df[[a1, a2]].dropna()
            both = both[(both[a1] != "") & (both[a2] != "")]
            if len(both) > 0:
                stats["annotator_agreement"] = round(
                    float((both[a1] == both[a2]).mean()), 4
                )
                stats["annotated_count"] = int(len(both))
        summary[model_name] = stats
    _write(HUMAN_EVAL_DIR / "summary.json", summary)

    flagged = {}
    for model_name, df in data_store.human_eval.items():
        sub = df[df["flag_reason"].notna() & (df["flag_reason"] != "")]
        flagged[model_name] = {
            "items": _clean(sub.to_dict(orient="records")),
            "total": int(len(sub)),
        }
    _write(HUMAN_EVAL_DIR / "flagged.json", flagged)

    _write(HUMAN_EVAL_DIR / "heldout.json", data_store.heldout)


def export_inference_models():
    """Static mode: both models are unavailable. Frontend renders a banner."""
    print("[inference]")
    _write(
        OUT_DIR / "inference-models.json",
        [
            {
                "name": "bart-ce-rl",
                "display": "BART CE+RL",
                "available": False,
                "loaded": False,
                "note": "Live inference requires running the backend locally",
            },
            {
                "name": "llama-3.2-1b",
                "display": "LLaMA 3.2 1B",
                "available": False,
                "loaded": False,
                "note": "Live inference requires running the backend + LMStudio locally",
            },
        ],
    )


def main():
    print(f"Exporting to {OUT_DIR.relative_to(PROJECT_ROOT)}/")
    print("Loading data store...")
    data_store.load()
    print(f"  {len(data_store.results)} model results, "
          f"{len(data_store.mislabels)} mislabels, "
          f"{len(data_store.heldout)} heldout records\n")

    export_metrics()
    export_samples()
    export_mislabels()
    export_multi_classifier()
    export_golden()
    export_human_eval()
    export_inference_models()

    total_size = sum(p.stat().st_size for p in OUT_DIR.rglob("*.json"))
    print(f"\nDone. Total: {total_size / 1024 / 1024:.1f} MB across "
          f"{sum(1 for _ in OUT_DIR.rglob('*.json'))} files.")


if __name__ == "__main__":
    main()
