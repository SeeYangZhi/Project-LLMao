from __future__ import annotations

import math
from typing import Optional

from fastapi import APIRouter, Query

from app.services.data_loader import data_store


def _clean_records(records: list[dict]) -> list[dict]:
    """Replace NaN/inf values with None for JSON serialization."""
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                rec[k] = None
    return records

router = APIRouter(prefix="/api/human-eval", tags=["human-eval"])


@router.get("/gold")
def get_gold(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    if data_store.gold_eval is None:
        return {"items": [], "total": 0}

    df = data_store.gold_eval
    total = len(df)
    start = (page - 1) * page_size
    page_df = df.iloc[start : start + page_size]
    return {
        "items": _clean_records(page_df.to_dict(orient="records")),
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/flagged")
def get_flagged(model: Optional[str] = Query(None)):
    results = {}
    models = [model] if model else list(data_store.human_eval.keys())
    for m in models:
        df = data_store.human_eval.get(m)
        if df is None:
            continue
        flagged = df[df["flag_reason"].notna() & (df["flag_reason"] != "")]
        results[m] = {
            "items": _clean_records(flagged.to_dict(orient="records")),
            "total": len(flagged),
        }
    return results


@router.get("/summary")
def get_summary():
    summary = {}
    for model_name, df in data_store.human_eval.items():
        stats = {
            "total_samples": len(df),
            "flagged_count": len(df[df["flag_reason"].notna() & (df["flag_reason"] != "")]),
        }
        if "suspicion_score" in df.columns:
            stats["mean_suspicion_score"] = round(df["suspicion_score"].mean(), 4)

        # Compute annotator agreement if columns exist
        a1_col = "annotator_1_sarcasm_removed"
        a2_col = "annotator_2_sarcasm_removed"
        if a1_col in df.columns and a2_col in df.columns:
            both = df[[a1_col, a2_col]].dropna()
            both = both[(both[a1_col] != "") & (both[a2_col] != "")]
            if len(both) > 0:
                agree = (both[a1_col] == both[a2_col]).mean()
                stats["annotator_agreement"] = round(agree, 4)
                stats["annotated_count"] = len(both)

        summary[model_name] = stats
    return summary


@router.get("/heldout")
def get_heldout(
    strategy: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    items = data_store.heldout
    if strategy:
        items = [r for r in items if r.get("strategy") == strategy]
    total = len(items)
    start = (page - 1) * page_size
    return {
        "items": items[start : start + page_size],
        "total": total,
        "page": page,
        "page_size": page_size,
    }
