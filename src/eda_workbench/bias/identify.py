"""Orchestrate bias identification.

Coverage runs on the profiled table. Fairlearn metrics and SHAP run only on
an explicit evaluation bundle. This module never trains a model and never
calls ``predict`` on the profiled dataset.

Paper (Section 6, ``sec:identification``):

- coverage → §6.3 Dataset and cohort bias auditing
  (``sec:dataset-and-cohort-bias-auditing``)
- metrics → §6.1 Metric-based fairness auditing
  (``sec:metric-based-fairness-auditing``); families in Section 5
- SHAP → §6.2 Explainability-driven bias diagnosis
  (``sec:explainability-driven-bias-diagnosis``). The paper describes this
  family via saliency / perturbation maps and does not name SHAP; this
  workbench uses tabular SHAP for sklearn models.

Full technique list by implementation lift: ``eda_workbench.bias.roadmap``
(and ``docs/bias-roadmap.md``).
"""

from __future__ import annotations

import pandas as pd

from eda_workbench.bias.artifacts import load_eval_bundle
from eda_workbench.bias.coverage import compute_coverage
from eda_workbench.bias.metric_audit import audit_metrics
from eda_workbench.bias.prepare import resolve_thresholds
from eda_workbench.bias.shap_audit import explain_eval
from eda_workbench.bias.types import (
    BiasIdentificationResult,
    MetricAuditResult,
    ShapAuditResult,
)


def _unique_reasons(reasons: list[str]) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason not in seen:
            unique.append(reason)
            seen.add(reason)
    return tuple(unique)


def identify_bias(
    frame: pd.DataFrame,
    dataset_config: dict,
    bias_config: dict | None = None,
    *,
    profiled_path: str | None = None,
    random_seed: int = 42,
) -> BiasIdentificationResult:
    """Run coverage, then metrics/SHAP on user-supplied eval artifacts.

    Paper: Section 6 identification pipeline (coverage §6.3, metrics §6.1,
    explainability family §6.2). Mitigation (Section 7) is out of scope.

    Args:
        frame: Profiled dataset. Used **only** for representational coverage.
        dataset_config: Dataset YAML including protected attributes and
            optional ``eval_path``, ``predictions_path``, ``model_path``.
        bias_config: Global ``bias:`` thresholds.
        profiled_path: Path of ``frame``, used to refuse eval/prediction files
            that resolve to the same CSV.
        random_seed: Sampling seed for SHAP.

    Returns:
        Structured findings. Missing artifacts produce skip reasons instead
        of scoring ``frame``.
    """
    thresholds = resolve_thresholds(bias_config)
    skip_reasons: list[str] = []

    coverage = compute_coverage(frame, dataset_config, thresholds)
    if coverage.skipped:
        skip_reasons.append(coverage.skipped)
    skip_reasons.extend(coverage.notes)

    bundle = load_eval_bundle(dataset_config, profiled_path=profiled_path)

    if bundle.metrics_skip or bundle.predictions is None:
        metrics = MetricAuditResult(
            skipped=bundle.metrics_skip
            or "Fairness metrics skipped: no predictions table loaded."
        )
        skip_reasons.append(metrics.skipped)
    else:
        try:
            metrics = audit_metrics(bundle.predictions, dataset_config, thresholds)
        except Exception as exc:
            metrics = MetricAuditResult(
                skipped=f"Fairness metrics skipped: unexpected error ({exc})."
            )
        if metrics.skipped:
            skip_reasons.append(metrics.skipped)
        skip_reasons.extend(metrics.notes)

    if bundle.shap_skip or bundle.model is None or bundle.eval_features is None:
        shap_result = ShapAuditResult(
            skipped=bundle.shap_skip
            or "SHAP skipped: eval table or model was not loaded."
        )
        skip_reasons.append(shap_result.skipped)
    else:
        try:
            shap_result = explain_eval(
                bundle.model,
                bundle.eval_features,
                dataset_config,
                thresholds,
                random_seed=random_seed,
            )
        except Exception as exc:
            shap_result = ShapAuditResult(
                skipped=f"SHAP skipped: unexpected error ({exc})."
            )
        if shap_result.skipped:
            skip_reasons.append(shap_result.skipped)
        skip_reasons.extend(shap_result.notes)

    return BiasIdentificationResult(
        coverage=coverage,
        metrics=metrics,
        shap=shap_result,
        skip_reasons=_unique_reasons(skip_reasons),
    )
