from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from app.services.data_loader import data_store

router = APIRouter(prefix="/api/mislabels", tags=["mislabels"])


@router.get("/summary")
def get_summary():
    items = data_store.mislabels
    over = sum(1 for m in items if m["direction"] == "over")
    under = sum(1 for m in items if m["direction"] == "under")
    onion = sum(1 for m in items if m["source"] == "theonion")
    huffpost = sum(1 for m in items if m["source"] == "huffpost")
    high_conf = sum(
        1 for m in items
        if m.get("stepfun_confidence") == "high" and m.get("nemotron_confidence") == "high"
    )
    return {
        "total": len(items),
        "over_labeled": over,      # NHDSD marked sarcastic; probably not
        "under_labeled": under,    # NHDSD marked non-sarcastic; probably is
        "theonion": onion,
        "huffpost": huffpost,
        "both_models_high_confidence": high_conf,
    }


@router.get("")
def get_mislabels(
    direction: Optional[str] = Query(None, description="'over' or 'under'"),
    source: Optional[str] = Query(None, description="'theonion' or 'huffpost'"),
    confidence: Optional[str] = Query(None, description="'high' to require both models high"),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(15, ge=1, le=100),
):
    items = data_store.mislabels

    if direction in ("over", "under"):
        items = [m for m in items if m["direction"] == direction]
    if source in ("theonion", "huffpost"):
        items = [m for m in items if m["source"] == source]
    if confidence == "high":
        items = [
            m for m in items
            if m.get("stepfun_confidence") == "high"
            and m.get("nemotron_confidence") == "high"
        ]
    if search:
        q = search.lower()
        items = [m for m in items if q in (m.get("headline") or "").lower()]

    total = len(items)
    start = (page - 1) * page_size
    return {
        "items": items[start : start + page_size],
        "total": total,
        "page": page,
        "page_size": page_size,
    }
