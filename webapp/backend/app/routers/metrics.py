from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from app.config import BASELINES, METRIC_COLUMNS, MODEL_REGISTRY
from app.services.data_loader import data_store

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/summary")
def get_summary():
    return {
        "models": data_store.summary,
        "baselines": BASELINES,
        "registry": MODEL_REGISTRY,
    }


@router.get("/by-strategy")
def get_by_strategy(model: Optional[str] = Query(None)):
    if model:
        return data_store.strategy_summary.get(model, {})
    return data_store.strategy_summary


@router.get("/distribution/{metric}")
def get_distribution(metric: str, model: Optional[str] = Query(None)):
    if metric not in METRIC_COLUMNS:
        return {"error": f"Unknown metric: {metric}"}

    result = {}
    models = [model] if model else list(data_store.results.keys())
    for m in models:
        df = data_store.results.get(m)
        if df is not None and metric in df.columns:
            values = df[metric].dropna().tolist()
            result[m] = values
    return result


@router.get("/models")
def get_models():
    available = list(data_store.results.keys())
    return [
        {
            "name": name,
            "display": MODEL_REGISTRY.get(name, {}).get("display", name),
            "type": MODEL_REGISTRY.get(name, {}).get("type", "unknown"),
            "sample_count": len(data_store.results[name]),
        }
        for name in available
    ]
