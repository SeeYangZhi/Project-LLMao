"""Load all result CSVs into memory at startup and compute aggregates."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from app.config import (
    CROSS_VAL_FILE,
    DATA_DIR,
    HELDOUT_FILE,
    HUMAN_EVAL_CSV,
    METRIC_COLUMNS,
    MODEL_REGISTRY,
    RESULTS_DIR,
    STRATEGIES,
)


def _to_native(d: dict) -> dict:
    """Convert numpy types to Python natives for JSON serialization."""
    import numpy as np
    return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in d.items()}


_THEONION_HOST_RE = re.compile(r"^https?://[a-z0-9-]+\.theonion\.com", re.IGNORECASE)


def _normalize_url(url: Optional[str]) -> Optional[str]:
    """Collapse every TheOnion subdomain (www., local., politics., etc.) to
    the canonical `theonion.com` host. Non-TheOnion URLs pass through
    unchanged."""
    if not url:
        return url
    return _THEONION_HOST_RE.sub("https://theonion.com", url)


class DataStore:
    def __init__(self):
        self.results: dict[str, pd.DataFrame] = {}
        self.human_eval: dict[str, pd.DataFrame] = {}
        self.gold_eval: Optional[pd.DataFrame] = None
        self.heldout: list[dict] = []
        self.mislabels: list[dict] = []
        self.summary: dict[str, dict[str, float]] = {}
        self.strategy_summary: dict[str, dict[str, dict[str, float]]] = {}

    def load(self):
        self._load_results()
        self._load_human_eval()
        self._load_gold_eval()
        self._load_heldout()
        self._load_mislabels()
        self._compute_summary()
        self._compute_strategy_summary()

    def _load_results(self):
        for model_name in MODEL_REGISTRY:
            path = RESULTS_DIR / f"{model_name}_results.csv"
            if path.exists():
                self.results[model_name] = pd.read_csv(path)

    def _load_human_eval(self):
        for model_name in MODEL_REGISTRY:
            path = RESULTS_DIR / f"{model_name}_results_human_eval.csv"
            if path.exists():
                self.human_eval[model_name] = pd.read_csv(path)

    def _load_gold_eval(self):
        if HUMAN_EVAL_CSV.exists():
            self.gold_eval = pd.read_csv(HUMAN_EVAL_CSV)

    def _load_heldout(self):
        if HELDOUT_FILE.exists():
            with open(HELDOUT_FILE) as f:
                self.heldout = [json.loads(line) for line in f]

    def _load_mislabels(self):
        """Load cross-validation records and keep only confirmed mislabels
        (original NHDSD disagrees, StepFun and Nemotron both agree)."""
        if not CROSS_VAL_FILE.exists():
            return
        with open(CROSS_VAL_FILE) as f:
            rows = [json.loads(line) for line in f]
        mislabels = []
        for idx, r in enumerate(rows):
            original = r.get("original_label")
            stepfun = r.get("stepfun_label")
            nemotron = r.get("is_sarcastic")  # field name from script
            if original is None or stepfun is None or nemotron is None:
                continue
            if original != stepfun and stepfun == nemotron:
                raw_link = r.get("article_link") or ""
                is_onion = "theonion.com" in raw_link
                article_link = _normalize_url(raw_link) if is_onion else raw_link
                mislabels.append({
                    "id": idx,
                    "headline": r.get("headline"),
                    "article_link": article_link,
                    "original_label": original,
                    "stepfun_label": stepfun,
                    "nemotron_label": nemotron,
                    "stepfun_confidence": r.get("stepfun_confidence"),
                    "nemotron_confidence": r.get("confidence"),
                    # Direction: "over" = NHDSD said sarcastic but it isn't;
                    # "under" = NHDSD said non-sarcastic but it is
                    "direction": "over" if original == 1 else "under",
                    "source": "theonion" if is_onion else "huffpost",
                })
        self.mislabels = mislabels

    def _compute_summary(self):
        for model_name, df in self.results.items():
            cols = [c for c in METRIC_COLUMNS if c in df.columns]
            self.summary[model_name] = _to_native(df[cols].mean().to_dict())
            if "hard_flipped" in df.columns:
                self.summary[model_name]["hard_flip_rate"] = float(df["hard_flipped"].mean() * 100)

    def _compute_strategy_summary(self):
        for model_name, df in self.results.items():
            if "subtype" not in df.columns:
                continue
            self.strategy_summary[model_name] = {}
            cols = [c for c in METRIC_COLUMNS if c in df.columns]
            for strategy in STRATEGIES:
                subset = df[df["subtype"] == strategy]
                if len(subset) > 0:
                    stats = _to_native(subset[cols].mean().to_dict())
                    if "hard_flipped" in subset.columns:
                        stats["hard_flip_rate"] = float(subset["hard_flipped"].mean() * 100)
                    stats["count"] = len(subset)
                    self.strategy_summary[model_name][strategy] = stats


data_store = DataStore()
