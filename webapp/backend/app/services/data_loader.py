"""Load all result CSVs into memory at startup and compute aggregates."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from app.config import (
    CLASSIFIER_RENAMES,
    CLASSIFIERS,
    CROSS_VAL_FILE,
    DATA_DIR,
    GOLDEN_CLF_SUMMARY_FILE,
    GOLDEN_DIR,
    GOLDEN_MODELS,
    GOLDEN_SUMMARY_FILE,
    HELDOUT_FILE,
    HUMAN_EVAL_CSV,
    METRIC_COLUMNS,
    MODEL_REGISTRY,
    MULTI_CLF_FILE,
    RESULTS_DIR,
    STRATEGIES,
)


def _rename_clf(value):
    """Apply the classifier rename map to a single string value."""
    return CLASSIFIER_RENAMES.get(value, value) if isinstance(value, str) else value


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
        # Multi-classifier eval (3 classifiers × 14 models)
        self.multi_classifier: dict[str, list[dict]] = {}
        # Golden human eval (3 models × 140 samples × 2 annotators)
        self.golden_summary: list[dict] = []
        self.golden_classifier_breakdown: list[dict] = []
        self.golden_subtype: dict[str, list[dict]] = {}
        self.golden_samples: dict[str, list[dict]] = {}

    def load(self):
        self._load_results()
        self._load_human_eval()
        self._load_gold_eval()
        self._load_heldout()
        self._load_mislabels()
        self._load_multi_classifier()
        self._load_golden()
        self._compute_summary()
        self._compute_strategy_summary()

    def _load_results(self):
        for model_name in MODEL_REGISTRY:
            path = RESULTS_DIR / f"{model_name}_results.csv"
            if path.exists():
                df = pd.read_csv(path)
                # Rename the legacy classifier column suffix so downstream
                # JSON / API consumers see the corrected identifier.
                df = df.rename(
                    columns={
                        c: c.replace("distilbert_reddit", "bert_kaggle")
                        for c in df.columns
                    }
                )
                self.results[model_name] = df

    def _load_human_eval(self):
        for model_name in MODEL_REGISTRY:
            path = RESULTS_DIR / f"{model_name}_results_human_eval.csv"
            if path.exists():
                df = pd.read_csv(path)
                df = df.rename(
                    columns={
                        c: c.replace("distilbert_reddit", "bert_kaggle")
                        for c in df.columns
                    }
                )
                self.human_eval[model_name] = df

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

    def _load_multi_classifier(self):
        """Per-model flip rates from all 3 classifiers (long format CSV).

        The eval pipeline labels the helinivan/english-sarcasm-detector
        rows as 'DistilBERT-Reddit' even though the model is actually a
        BERT trained on a Kaggle headlines dataset; we rename here so
        every downstream JSON / page uses the corrected names.
        """
        if not MULTI_CLF_FILE.exists():
            return
        df = pd.read_csv(MULTI_CLF_FILE)
        for model_name, sub in df.groupby("model"):
            self.multi_classifier[model_name] = [
                {
                    "classifier": _rename_clf(row["classifier"]),
                    "type": _rename_clf(row["type"]),
                    "training": _rename_clf(row["training"]),
                    "flip_rate": float(row["flip_rate"]),
                    "mean_flip_delta": float(row["mean_flip_delta"]),
                }
                for _, row in sub.iterrows()
            ]

    def _load_golden(self):
        """Load human-annotated golden eval (3 models, 140 samples each).

        We compute summary stats from each model's `_merged.csv` directly
        rather than trusting `summary.csv` — the upstream pipeline rounds
        meaning_change_rate to one decimal in the summary file, while the
        merged CSVs hold the exact per-row labels.
        """
        if GOLDEN_CLF_SUMMARY_FILE.exists():
            df = pd.read_csv(GOLDEN_CLF_SUMMARY_FILE)
            records = []
            for r in df.to_dict(orient="records"):
                rec = _to_native(r)
                # Rename the classifier label columns
                if "classifier" in rec:
                    rec["classifier"] = _rename_clf(rec["classifier"])
                if "classifier_type" in rec:
                    rec["classifier_type"] = _rename_clf(rec["classifier_type"])
                if "training_data" in rec:
                    rec["training_data"] = _rename_clf(rec["training_data"])
                records.append(rec)
            self.golden_classifier_breakdown = records
        for model_key in GOLDEN_MODELS:
            merged_path = GOLDEN_DIR / f"{model_key}_merged.csv"
            if merged_path.exists():
                df = pd.read_csv(merged_path)
                # Compute exact summary from row-level labels
                stats = {
                    "model": model_key,
                    "display": GOLDEN_MODELS[model_key],
                    "n_samples": len(df),
                    "human_flip_rate": float(df["human_flipped_strict"].mean()),
                    "meaning_change_rate": float(df["meaning_change"].mean()),
                    "strict_success_rate": float(df["human_strict_success"].mean()),
                    "mean_similarity": float(df["similarity"].mean()),
                    "mean_edit_dist": float(df["edit_dist_norm"].mean()),
                    "mean_paraphrase": float(df["paraphrase_score"].mean()),
                }
                # Inter-annotator κ comes from summary.csv (already correct)
                if GOLDEN_SUMMARY_FILE.exists():
                    sdf = pd.read_csv(GOLDEN_SUMMARY_FILE)
                    row = sdf[sdf["model"] == model_key]
                    if len(row) > 0:
                        stats["inter_annotator_kappa"] = float(
                            row.iloc[0]["inter_annotator_kappa"]
                        )
                self.golden_summary.append(stats)

                # Rename the classifier columns up-front so downstream JSON
                # uses the corrected identifier
                df = df.rename(
                    columns={
                        c: c.replace("distilbert_reddit", "bert_kaggle")
                        for c in df.columns
                    }
                )
                # Keep only the columns the frontend needs for sample browsing
                keep = [
                    "id", "input", "output", "subtype",
                    "human_flipped_strict", "meaning_change", "human_strict_success",
                    "human_meaning_preserved",
                    "hard_flipped_roberta_twitter", "hard_flipped_bert_kaggle",
                    "hard_flipped_roberta_news", "similarity", "edit_dist_norm",
                    "paraphrase_score", "error_type",
                ]
                cols = [c for c in keep if c in df.columns]
                self.golden_samples[model_key] = df[cols].where(
                    df[cols].notna(), None
                ).to_dict(orient="records")

            sub_path = GOLDEN_DIR / f"{model_key}_subtype_analysis.csv"
            if sub_path.exists():
                df = pd.read_csv(sub_path)
                # Rename reddit_* columns to kaggle_* to match the corrected
                # classifier identifier
                df = df.rename(
                    columns={c: c.replace("reddit_", "kaggle_") for c in df.columns}
                )
                self.golden_subtype[model_key] = [
                    _to_native(r) for r in df.to_dict(orient="records")
                ]

    def _compute_summary(self):
        for model_name, df in self.results.items():
            cols = [c for c in METRIC_COLUMNS if c in df.columns]
            self.summary[model_name] = _to_native(df[cols].mean().to_dict())
            if "hard_flipped" in df.columns:
                self.summary[model_name]["hard_flip_rate"] = float(df["hard_flipped"].mean() * 100)
            # Per-classifier flip rates so the dashboard can show all 3.
            # Column names already renamed in _load_results to use the
            # corrected classifier identifier.
            for col, key in (
                ("hard_flipped_roberta_twitter", "flip_rate_twitter"),
                ("hard_flipped_bert_kaggle", "flip_rate_kaggle"),
                ("hard_flipped_roberta_news", "flip_rate_news"),
            ):
                if col in df.columns:
                    self.summary[model_name][key] = float(df[col].mean() * 100)
            # Spread between best and worst classifier — illustrates how
            # unreliable any single flip rate is.
            clf_rates = [
                self.summary[model_name].get(k)
                for k in ("flip_rate_twitter", "flip_rate_kaggle", "flip_rate_news")
                if self.summary[model_name].get(k) is not None
            ]
            if len(clf_rates) >= 2:
                self.summary[model_name]["flip_rate_spread"] = max(clf_rates) - min(clf_rates)
            # Median perplexity — the mean is dominated by long-tail outliers
            if "perplexity" in df.columns:
                self.summary[model_name]["perplexity_median"] = float(df["perplexity"].median())

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
