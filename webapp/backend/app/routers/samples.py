from __future__ import annotations

import math
from typing import Optional

from fastapi import APIRouter, Query

from app.services.data_loader import data_store


def _clean_records(records: list[dict]) -> list[dict]:
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                rec[k] = None
    return records

router = APIRouter(prefix="/api/samples", tags=["samples"])


@router.get("")
def get_samples(
    model: str = Query("bart_base_ce_rl"),
    strategy: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    sort_by: str = Query("id"),
    sort_order: str = Query("asc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    df = data_store.results.get(model)
    if df is None:
        return {"items": [], "total": 0, "page": page, "page_size": page_size}

    filtered = df.copy()

    if strategy:
        strategies = strategy.split(",")
        filtered = filtered[filtered["subtype"].isin(strategies)]

    if search:
        mask = filtered["input"].str.contains(search, case=False, na=False) | \
               filtered["output"].str.contains(search, case=False, na=False)
        filtered = mask_df(filtered, mask)

    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=(sort_order == "asc"))

    total = len(filtered)
    start = (page - 1) * page_size
    page_df = filtered.iloc[start : start + page_size]

    return {
        "items": _clean_records(page_df.to_dict(orient="records")),
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def mask_df(df, mask):
    return df[mask]


@router.get("/{sample_id}/compare")
def compare_sample(sample_id: int):
    result = {}
    for model_name, df in data_store.results.items():
        row = df[df["id"] == sample_id]
        if len(row) > 0:
            result[model_name] = row.iloc[0].to_dict()
    return result
