"""Structured results for bias identification (no Markdown)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroupShare:
    """Per-group count and share for one protected attribute."""

    attribute: str
    group: str
    count: int
    share: float
    underrepresented: bool


@dataclass(frozen=True)
class AttributeCoverage:
    """Coverage summary for a single protected attribute."""

    attribute: str
    n_missing: int
    imbalance_ratio: float
    groups: tuple[GroupShare, ...]


@dataclass(frozen=True)
class IntersectionShare:
    """Count and share for one combination of protected-attribute values."""

    combination: tuple[tuple[str, str], ...]
    count: int
    share: float
    sparse: bool


@dataclass(frozen=True)
class CoverageResult:
    """Representational coverage of the profiled table.

    Paper: §6.3 ``sec:dataset-and-cohort-bias-auditing``.
    """

    n_rows: int
    attributes: tuple[AttributeCoverage, ...] = ()
    intersections: tuple[IntersectionShare, ...] = ()
    notes: tuple[str, ...] = ()
    skipped: str | None = None


@dataclass(frozen=True)
class GroupMetrics:
    """Fairlearn per-group metrics for one protected attribute value.

    selection_rate: §5.1; TPR/FPR: §5.2; precision: §5.3; accuracy: §5.5;
    roc_auc: §5.4 (optional, needs scores).
    """

    attribute: str
    group: str
    count: int
    selection_rate: float | None = None
    true_positive_rate: float | None = None
    false_positive_rate: float | None = None
    precision: float | None = None
    accuracy: float | None = None
    roc_auc: float | None = None


@dataclass(frozen=True)
class AttributeMetricSummary:
    """Cross-group fairness summaries and threshold flags for one attribute.

    Paper: §6.1 dashboard using §5.1 DP difference/ratio, §5.2 equal
    opportunity / equalized odds, §5.5 worst-group accuracy and accuracy gap.
    """

    attribute: str
    demographic_parity_difference: float | None = None
    demographic_parity_ratio: float | None = None
    equal_opportunity_difference: float | None = None
    equalized_odds_difference: float | None = None
    worst_group_accuracy: float | None = None
    accuracy_gap: float | None = None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricAuditResult:
    """Fairlearn dashboard computed from provided eval predictions only.

    Paper: §6.1 ``sec:metric-based-fairness-auditing``.
    """

    n_rows: int = 0
    binary_target: bool = False
    by_group: tuple[GroupMetrics, ...] = ()
    summaries: tuple[AttributeMetricSummary, ...] = ()
    notes: tuple[str, ...] = ()
    skipped: str | None = None


@dataclass(frozen=True)
class ShapFeatureImportance:
    """Global mean absolute SHAP value for one feature."""

    feature: str
    mean_abs_shap: float


@dataclass(frozen=True)
class ShapGroupMean:
    """Mean SHAP value for one feature inside one protected group."""

    attribute: str
    group: str
    feature: str
    mean_shap: float


@dataclass(frozen=True)
class ShapAuditResult:
    """SHAP attributions computed on eval rows only.

    Paper: §6.2 family ``sec:explainability-driven-bias-diagnosis`` (SHAP
    is not named in the paper).
    """

    n_explained: int = 0
    global_importance: tuple[ShapFeatureImportance, ...] = ()
    by_group: tuple[ShapGroupMean, ...] = ()
    flags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    skipped: str | None = None


@dataclass(frozen=True)
class BiasIdentificationResult:
    """Combined coverage, metrics, and SHAP findings.

    Paper: Section 6 ``sec:identification`` (not Section 7 mitigation).
    """

    coverage: CoverageResult
    metrics: MetricAuditResult
    shap: ShapAuditResult
    skip_reasons: tuple[str, ...] = ()

    @property
    def ran_metrics(self) -> bool:
        return self.metrics.skipped is None

    @property
    def ran_shap(self) -> bool:
        return self.shap.skipped is None
