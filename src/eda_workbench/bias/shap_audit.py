"""SHAP bias diagnosis on the evaluation feature table only.

``predict`` / ``predict_proba`` / ``shap.Explainer`` may run on ``eval_features``
only — never on the profiled dataset CSV.

Paper: Section 6.2 Explainability-driven bias diagnosis
(``sec:explainability-driven-bias-diagnosis``). The paper describes this
family as evidence of what the model uses (saliency / perturbation maps,
intermediate representations). SHAP is not named; this module is the
tabular sklearn instantiation of that family, not imaging saliency.

``PermutationExplainer`` runs on a numeric view of the eval table: string
columns are mapped to codes so SHAP's masker can compare values, then decoded
before ``predict`` / ``predict_proba``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from eda_workbench.bias.prepare import (
    as_name_list,
    group_labels,
    protected_attributes,
    split_present_attributes,
)
from eda_workbench.bias.types import (
    ShapAuditResult,
    ShapFeatureImportance,
    ShapGroupMean,
)


def resolve_feature_names(model: Any, dataset_config: dict) -> list[str] | None:
    """Feature columns from ``feature_names_in_`` or YAML ``model_features``."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]

    named_steps = getattr(model, "named_steps", None)
    if named_steps:
        for step in named_steps.values():
            step_names = getattr(step, "feature_names_in_", None)
            if step_names is not None:
                return [str(name) for name in step_names]

    configured = dataset_config.get("model_features") or None
    if configured:
        return as_name_list(configured)
    return None


def _as_2d(values: object) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim > 2:
        raise ValueError(
            "SHAP values have a class axis; this workbench reports a single output only."
        )
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _category_maps(
    frame: pd.DataFrame, feature_names: list[str]
) -> dict[str, tuple[dict[str, int], dict[int, str]]]:
    """Map non-numeric columns to integer codes (SHAP maskers need numbers)."""
    maps: dict[str, tuple[dict[str, int], dict[int, str]]] = {}
    for name in feature_names:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        ):
            continue
        labels = series.astype("string")
        uniques = [str(value) for value in pd.unique(labels)]
        to_code = {value: index for index, value in enumerate(uniques)}
        to_label = {index: value for value, index in to_code.items()}
        maps[name] = (to_code, to_label)
    return maps


def _to_numeric_view(
    frame: pd.DataFrame,
    feature_names: list[str],
    maps: dict[str, tuple[dict[str, int], dict[int, str]]],
) -> pd.DataFrame:
    encoded = frame.loc[:, feature_names].copy()
    for name in feature_names:
        if name in maps:
            to_code, _ = maps[name]
            encoded[name] = encoded[name].astype("string").map(to_code).astype(float)
        else:
            encoded[name] = pd.to_numeric(encoded[name], errors="coerce")
    return encoded


def _decode_numeric_view(
    frame: pd.DataFrame,
    feature_names: list[str],
    maps: dict[str, tuple[dict[str, int], dict[int, str]]],
) -> pd.DataFrame:
    decoded = frame.loc[:, feature_names].copy()
    for name, (_, to_label) in maps.items():
        max_code = max(to_label) if to_label else 0

        def _one(code: object, labels: dict[int, str] = to_label, cap: int = max_code):
            if code is None or (isinstance(code, float) and np.isnan(code)):
                raw = labels.get(0, pd.NA)
            else:
                key = int(round(float(code)))
                key = min(max(key, 0), cap)
                raw = labels.get(key, labels[cap])
            return pd.NA if raw == "<NA>" else raw

        decoded[name] = [_one(value) for value in decoded[name].to_numpy()]
    return decoded


def _predict_fn(
    model: Any,
    feature_names: list[str],
    maps: dict[str, tuple[dict[str, int], dict[int, str]]],
):
    def predict_proba_or_label(data):
        if isinstance(data, pd.DataFrame):
            numeric = data.loc[:, feature_names]
        else:
            numeric = pd.DataFrame(data, columns=feature_names)
        features = _decode_numeric_view(numeric, feature_names, maps)
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(features))
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        return np.asarray(model.predict(features))

    return predict_proba_or_label


def explain_eval(
    model: Any,
    eval_features: pd.DataFrame,
    dataset_config: dict,
    thresholds: dict,
    *,
    random_seed: int = 42,
) -> ShapAuditResult:
    """Explain sampled rows from the eval table with ``shap.Explainer``.

    Paper: §6.2 family (``sec:explainability-driven-bias-diagnosis``).
    Group-stratified mean SHAP is how this workbench localizes attribution
    differences across protected groups.

    Args:
        model: Sklearn-compatible estimator with ``predict``.
        eval_features: Evaluation feature table (never the profiled CSV).
        dataset_config: Protected attributes and optional ``model_features``.
        thresholds: SHAP sample sizes and attribution-gap flag.
        random_seed: Sampling seed.
    """
    feature_names = resolve_feature_names(model, dataset_config)
    if feature_names is None:
        return ShapAuditResult(
            skipped=(
                "SHAP skipped: cannot determine model feature columns "
                "(no feature_names_in_ or model_features)."
            )
        )

    missing_features = [
        name for name in feature_names if name not in eval_features.columns
    ]
    if missing_features:
        listed = ", ".join(missing_features)
        return ShapAuditResult(
            skipped=f"SHAP skipped: eval table is missing model features: {listed}."
        )

    n_rows = len(eval_features)
    if n_rows == 0:
        return ShapAuditResult(skipped="SHAP skipped: eval feature table is empty.")

    n_background = max(1, min(int(thresholds["shap_background_size"]), n_rows))
    n_explain = max(1, min(int(thresholds["shap_explain_size"]), n_rows))
    background = eval_features.sample(n=n_background, random_state=random_seed)
    explained = eval_features.sample(n=n_explain, random_state=random_seed)
    maps = _category_maps(eval_features, feature_names)
    x_background = _to_numeric_view(background, feature_names, maps)
    x_explain = _to_numeric_view(explained, feature_names, maps)
    predict_fn = _predict_fn(model, feature_names, maps)
    probe = np.asarray(predict_fn(x_background.iloc[:1]))
    if probe.ndim == 2 and probe.shape[1] > 1:
        return ShapAuditResult(
            skipped=(
                "SHAP skipped: model output is multiclass. "
                "This workbench reports attributions for a single output only."
            )
        )

    explainer = shap.PermutationExplainer(
        predict_fn,
        x_background,
        seed=random_seed,
    )
    explanation = explainer(x_explain)
    try:
        values = _as_2d(explanation.values)
    except ValueError:
        return ShapAuditResult(
            skipped=(
                "SHAP skipped: attributions have a class axis (multiclass "
                "output). This workbench reports a single output only."
            )
        )
    mean_abs = np.abs(values).mean(axis=0)
    importance = tuple(
        sorted(
            (
                ShapFeatureImportance(feature=name, mean_abs_shap=float(score))
                for name, score in zip(feature_names, mean_abs, strict=True)
            ),
            key=lambda item: item.mean_abs_shap,
            reverse=True,
        )
    )

    attributes = protected_attributes(dataset_config)
    present, missing = split_present_attributes(explained, attributes)
    notes: list[str] = []
    if missing:
        listed = ", ".join(f"`{name}`" for name in missing)
        notes.append(
            "Protected attributes not found in the eval table "
            f"(group SHAP skipped for them): {listed}."
        )

    by_group: list[ShapGroupMean] = []
    flags: list[str] = []
    gap_thr = float(thresholds["shap_attribution_gap_threshold"])
    for attr in present:
        labels = group_labels(explained[attr])
        for feature_idx, feature_name in enumerate(feature_names):
            means: dict[str, float] = {}
            for group in labels.unique():
                mask = (labels == group).to_numpy()
                if not mask.any():
                    continue
                mean_val = float(values[mask, feature_idx].mean())
                means[str(group)] = mean_val
                by_group.append(
                    ShapGroupMean(
                        attribute=attr,
                        group=str(group),
                        feature=feature_name,
                        mean_shap=mean_val,
                    )
                )
            if len(means) >= 2:
                gap = max(means.values()) - min(means.values())
                if gap > gap_thr:
                    flags.append(
                        f"{attr}/{feature_name}: cross-group mean SHAP gap "
                        f"{gap:.3f} exceeds {gap_thr}"
                    )

    return ShapAuditResult(
        n_explained=len(x_explain),
        global_importance=importance,
        by_group=tuple(by_group),
        flags=tuple(flags),
        notes=tuple(notes),
    )
