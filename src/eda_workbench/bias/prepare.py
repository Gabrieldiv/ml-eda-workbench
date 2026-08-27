"""Shared helpers for protected attributes, paths, and bias thresholds."""

from __future__ import annotations

import math

import pandas as pd

DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "min_group_count": 30,
    "min_group_share": 0.05,
    "sparse_intersection_share": 0.01,
    "rate_difference_threshold": 0.1,
    "disparate_impact_threshold": 0.8,
    "shap_background_size": 50,
    "shap_explain_size": 100,
    "shap_attribution_gap_threshold": 0.1,
}

MISSING_GROUP_LABEL = "<missing>"


def as_name_list(value: object) -> list[str]:
    """Normalize YAML names to a list without splitting a string into characters."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    names: list[str] = []
    try:
        items = list(value)
    except TypeError:
        text = str(value).strip()
        return [text] if text else []
    for item in items:
        text = str(item).strip()
        if text:
            names.append(text)
    return names


def protected_attributes(dataset_config: dict) -> list[str]:
    """Return configured protected-attribute column names."""
    return as_name_list(dataset_config.get("protected_attributes"))


def split_present_attributes(
    frame: pd.DataFrame, attributes: list[str]
) -> tuple[list[str], list[str]]:
    """Split attribute names into those present in ``frame`` and those missing."""
    present = [name for name in attributes if name in frame.columns]
    missing = [name for name in attributes if name not in frame.columns]
    return present, missing


def group_labels(series: pd.Series) -> pd.Series:
    """Cast a protected-attribute column to string labels, filling NA."""
    return series.astype("string").fillna(MISSING_GROUP_LABEL)


def resolve_thresholds(bias_config: dict | None) -> dict[str, float | int]:
    """Merge user bias config onto defaults."""
    merged = dict(DEFAULT_THRESHOLDS)
    if bias_config:
        for key, value in bias_config.items():
            if key in merged and value is not None:
                merged[key] = value
    return merged


def configured_path(value: object) -> str | None:
    """Return a filesystem path string, or None when unset."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    return text


def y_true_column(columns: pd.Index, dataset_config: dict) -> str | None:
    """Resolve the eval label column: ``y_true`` or the configured target name."""
    if "y_true" in columns:
        return "y_true"
    target = dataset_config.get("target_column")
    if target and target in columns:
        return str(target)
    return None


def prediction_column(dataset_config: dict) -> str:
    """Column holding precomputed predictions (default ``y_pred``)."""
    return str(dataset_config.get("prediction_column") or "y_pred")


def score_column(dataset_config: dict) -> str:
    """Optional column holding prediction scores (default ``y_score``)."""
    return str(dataset_config.get("score_column") or "y_score")


def imbalance_ratio(counts: pd.Series) -> float:
    """Return max/min positive group count; NaN when there are no groups."""
    positive = counts[counts > 0]
    if positive.empty:
        return math.nan
    return float(positive.max() / positive.min())
