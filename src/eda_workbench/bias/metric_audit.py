"""Fairlearn metrics on precomputed eval predictions only.

This module never calls ``predict`` / ``predict_proba``. Labels and scores
must already be present in the predictions table.

Paper: Section 6.1 Metric-based fairness auditing
(``sec:metric-based-fairness-auditing``) — the multi-metric dashboard.
Families come from Section 5 (``sec:metrics``): independence §5.1,
separation §5.2, sufficiency §5.3, group-aggregate / worst-case §5.5.
Per-group AUC, when ``y_score`` is present, is the ROC family §5.4.
"""

from __future__ import annotations

import math
from functools import partial

import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

from eda_workbench.bias.prepare import (
    group_labels,
    prediction_column,
    protected_attributes,
    score_column,
    split_present_attributes,
    y_true_column,
)
from eda_workbench.bias.types import (
    AttributeMetricSummary,
    GroupMetrics,
    MetricAuditResult,
)

_precision = partial(precision_score, zero_division=0)


def actual_positive_rate(y_true, y_pred) -> float:
    """Prevalence of the positive label in ``y_true`` (ignores ``y_pred``)."""
    return float(selection_rate(y_true, y_true))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _is_binary(series: pd.Series) -> bool:
    return len(pd.unique(series.dropna())) == 2


def _safe_fairness_metric(fn, y_true, y_pred, sensitive) -> float | None:
    try:
        return _optional_float(fn(y_true, y_pred, sensitive_features=sensitive))
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _group_auc(
    y_true: pd.Series, y_score: pd.Series, labels: pd.Series
) -> dict[str, float | None]:
    auc_by_group: dict[str, float | None] = {}
    for group in labels.unique():
        mask = labels == group
        truth = y_true[mask]
        scores = y_score[mask]
        valid = truth.notna() & scores.notna()
        truth = truth[valid]
        scores = scores[valid]
        if len(truth) == 0 or truth.nunique() < 2:
            auc_by_group[str(group)] = None
            continue
        try:
            auc_by_group[str(group)] = _optional_float(roc_auc_score(truth, scores))
        except ValueError:
            auc_by_group[str(group)] = None
    return auc_by_group


def _binary_metrics() -> dict:
    # Paper §5 metric families used in the §6.1 dashboard:
    # independence §5.1, separation §5.2, sufficiency §5.3, aggregate §5.5.
    return {
        "count": count,
        "actual_positive_rate": actual_positive_rate,
        "selection_rate": selection_rate,  # §5.1 independence (selection / DP)
        "true_positive_rate": true_positive_rate,  # §5.2 equal opportunity
        "false_positive_rate": false_positive_rate,  # §5.2 predictive equality
        "precision": _precision,  # §5.3 predictive parity (PPV)
        "accuracy": accuracy_score,  # §5.5 group-aggregate
    }


def _aggregate_metrics() -> dict:
    # Non-binary targets: only §5.5 group-aggregate (overall accuracy equality).
    return {
        "count": count,
        "accuracy": accuracy_score,
    }


def audit_metrics(
    predictions: pd.DataFrame,
    dataset_config: dict,
    thresholds: dict,
) -> MetricAuditResult:
    """Compute a multi-metric fairness dashboard from provided labels.

    Paper: §6.1 (``sec:metric-based-fairness-auditing``). Does not implement
    proxy association tests or threshold-free KS / d′ from the same section.

    Args:
        predictions: Eval predictions table (not the profiled CSV).
        dataset_config: Column names and protected attributes.
        thresholds: Resolved bias thresholds for flags.
    """
    attributes = protected_attributes(dataset_config)
    if not attributes:
        return MetricAuditResult(
            skipped=("Fairness metrics skipped: no protected attributes configured.")
        )

    label_col = y_true_column(predictions.columns, dataset_config)
    pred_col = prediction_column(dataset_config)
    if label_col is None:
        return MetricAuditResult(
            skipped=(
                "Fairness metrics skipped: no y_true (or target) column "
                "in the predictions file."
            )
        )
    if pred_col not in predictions.columns:
        return MetricAuditResult(
            skipped=(
                f"Fairness metrics skipped: prediction column `{pred_col}` "
                "not in the predictions file."
            )
        )

    present, missing = split_present_attributes(predictions, attributes)
    notes: list[str] = []
    if missing:
        listed = ", ".join(f"`{name}`" for name in missing)
        notes.append(
            f"Protected attributes not found in the predictions file: {listed}."
        )
    if not present:
        return MetricAuditResult(
            notes=tuple(notes),
            skipped=(
                "Fairness metrics skipped: none of the configured protected "
                "attributes are in the predictions file."
            ),
        )

    work = predictions.dropna(subset=[label_col, pred_col])
    if work.empty:
        return MetricAuditResult(
            notes=tuple(notes),
            skipped="Fairness metrics skipped: no rows with both y_true and y_pred.",
        )

    y_true = work[label_col]
    y_pred = work[pred_col]
    binary = _is_binary(y_true)
    metric_fns = _binary_metrics() if binary else _aggregate_metrics()
    rate_thr = float(thresholds["rate_difference_threshold"])
    di_thr = float(thresholds["disparate_impact_threshold"])

    score_col = score_column(dataset_config)
    has_scores = score_col in work.columns

    by_group: list[GroupMetrics] = []
    summaries: list[AttributeMetricSummary] = []

    for attr in present:
        labels = group_labels(work[attr])
        try:
            frame = MetricFrame(
                metrics=metric_fns,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=labels,
            )
        except ValueError as exc:
            notes.append(f"Could not compute MetricFrame for `{attr}`: {exc}.")
            continue

        auc_by_group: dict[str, float | None] = {}
        if binary and has_scores:
            auc_by_group = _group_auc(y_true, work[score_col], labels)

        grouped = frame.by_group
        if grouped.index.name is None:
            grouped = grouped.rename_axis("group")
        group_col = grouped.index.name
        grouped_rows = grouped.reset_index()
        accuracies: list[float] = []
        for record in grouped_rows.to_dict(orient="records"):
            group_value = record[group_col]
            accuracy = _optional_float(record.get("accuracy"))
            if accuracy is not None:
                accuracies.append(accuracy)
            by_group.append(
                GroupMetrics(
                    attribute=attr,
                    group=str(group_value),
                    count=int(record.get("count", 0) or 0),
                    actual_positive_rate=_optional_float(
                        record.get("actual_positive_rate")
                    ),
                    selection_rate=_optional_float(record.get("selection_rate")),
                    true_positive_rate=_optional_float(
                        record.get("true_positive_rate")
                    ),
                    false_positive_rate=_optional_float(
                        record.get("false_positive_rate")
                    ),
                    precision=_optional_float(record.get("precision")),
                    accuracy=accuracy,
                    roc_auc=auc_by_group.get(str(group_value)),
                )
            )

        dp_diff = eo_diff = odds_diff = dp_ratio = None
        if binary:
            # §5.1 independence: Demographic Parity (difference and ratio / DI).
            dp_diff = _safe_fairness_metric(
                demographic_parity_difference, y_true, y_pred, labels
            )
            dp_ratio = _safe_fairness_metric(
                demographic_parity_ratio, y_true, y_pred, labels
            )
            # §5.2 separation: Equal Opportunity (TPR) and Equalized Odds.
            eo_diff = _safe_fairness_metric(
                equal_opportunity_difference, y_true, y_pred, labels
            )
            odds_diff = _safe_fairness_metric(
                equalized_odds_difference, y_true, y_pred, labels
            )

        # §5.5 group-aggregate / worst-case.
        worst_acc = min(accuracies) if accuracies else None
        acc_gap = (max(accuracies) - min(accuracies)) if accuracies else None

        flags: list[str] = []
        if dp_diff is not None and abs(dp_diff) > rate_thr:
            flags.append(
                f"demographic parity difference {dp_diff:.3f} exceeds {rate_thr}"
            )
        if dp_ratio is not None and dp_ratio < di_thr:
            flags.append(f"demographic parity ratio {dp_ratio:.3f} is below {di_thr}")
        if eo_diff is not None and abs(eo_diff) > rate_thr:
            flags.append(
                f"equal opportunity difference {eo_diff:.3f} exceeds {rate_thr}"
            )
        if odds_diff is not None and abs(odds_diff) > rate_thr:
            flags.append(
                f"equalized odds difference {odds_diff:.3f} exceeds {rate_thr}"
            )
        if acc_gap is not None and acc_gap > rate_thr:
            flags.append(f"accuracy gap {acc_gap:.3f} exceeds {rate_thr}")

        summaries.append(
            AttributeMetricSummary(
                attribute=attr,
                demographic_parity_difference=dp_diff,
                demographic_parity_ratio=dp_ratio,
                equal_opportunity_difference=eo_diff,
                equalized_odds_difference=odds_diff,
                worst_group_accuracy=worst_acc,
                accuracy_gap=acc_gap,
                flags=tuple(flags),
            )
        )

    if not summaries:
        return MetricAuditResult(
            notes=tuple(notes),
            skipped="Fairness metrics skipped: no group metrics could be computed.",
        )

    return MetricAuditResult(
        n_rows=len(work),
        binary_target=binary,
        by_group=tuple(by_group),
        summaries=tuple(summaries),
        notes=tuple(notes),
    )
