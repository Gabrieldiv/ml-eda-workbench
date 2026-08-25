"""Reporting: Assumptions block when metrics/SHAP ran; skip text otherwise."""

from eda_workbench.bias.types import (
    AttributeCoverage,
    AttributeMetricSummary,
    BiasIdentificationResult,
    CoverageResult,
    GroupMetrics,
    GroupShare,
    IntersectionShare,
    MetricAuditResult,
    ShapAuditResult,
    ShapFeatureImportance,
    ShapGroupMean,
)
from eda_workbench.reporting.bias import (
    ASSUMPTIONS_HEADING,
    normalize_report_detail,
    render_bias_identification,
)


def _coverage_ok() -> CoverageResult:
    return CoverageResult(n_rows=10, skipped=None)


def _populated_result() -> BiasIdentificationResult:
    return BiasIdentificationResult(
        coverage=CoverageResult(
            n_rows=10,
            attributes=(
                AttributeCoverage(
                    attribute="sex",
                    n_missing=0,
                    imbalance_ratio=4.0,
                    groups=(
                        GroupShare("sex", "f", 8, 0.8, False),
                        GroupShare("sex", "m", 2, 0.2, True),
                    ),
                ),
            ),
            intersections=(
                IntersectionShare(
                    combination=(("sex", "f"), ("pclass", "1")),
                    count=1,
                    share=0.1,
                    sparse=True,
                ),
            ),
        ),
        metrics=MetricAuditResult(
            n_rows=10,
            binary_target=True,
            by_group=(
                GroupMetrics(
                    attribute="sex",
                    group="f",
                    count=8,
                    actual_positive_rate=0.6,
                    selection_rate=0.5,
                    true_positive_rate=0.6,
                    false_positive_rate=0.2,
                    precision=0.7,
                    accuracy=0.8,
                    roc_auc=0.9,
                ),
                GroupMetrics(
                    attribute="sex",
                    group="m",
                    count=2,
                    actual_positive_rate=0.2,
                    selection_rate=0.1,
                    true_positive_rate=0.4,
                    false_positive_rate=0.1,
                    precision=0.5,
                    accuracy=0.6,
                    roc_auc=0.7,
                ),
            ),
            summaries=(
                AttributeMetricSummary(
                    attribute="sex",
                    demographic_parity_difference=0.4,
                    demographic_parity_ratio=0.5,
                    equal_opportunity_difference=0.2,
                    equalized_odds_difference=0.3,
                    worst_group_accuracy=0.5,
                    accuracy_gap=0.2,
                    flags=("demographic parity difference 0.400 exceeds 0.1",),
                ),
            ),
        ),
        shap=ShapAuditResult(
            n_explained=8,
            global_importance=(ShapFeatureImportance("f1", 0.2),),
            by_group=(
                ShapGroupMean("sex", "f", "f1", 0.1),
                ShapGroupMean("sex", "m", "f1", -0.1),
            ),
            flags=("sex/f1: cross-group mean SHAP gap 0.200 exceeds 0.1",),
        ),
    )


def test_assumptions_present_when_metrics_ran():
    result = BiasIdentificationResult(
        coverage=_coverage_ok(),
        metrics=MetricAuditResult(n_rows=10, binary_target=True),
        shap=ShapAuditResult(
            skipped=(
                "SHAP skipped: eval_path is not configured "
                "(refusing to use the profiled dataset)."
            )
        ),
    )
    markdown = render_bias_identification(result)
    assert ASSUMPTIONS_HEADING in markdown
    assert "held-out" in markdown
    assert "profiled table's composition" in markdown


def test_assumptions_present_when_shap_ran():
    result = BiasIdentificationResult(
        coverage=_coverage_ok(),
        metrics=MetricAuditResult(
            skipped="Fairness metrics skipped: no predictions_path configured."
        ),
        shap=ShapAuditResult(n_explained=8),
    )
    markdown = render_bias_identification(result)
    assert ASSUMPTIONS_HEADING in markdown


def test_assumptions_absent_when_only_coverage_ran():
    result = BiasIdentificationResult(
        coverage=_coverage_ok(),
        metrics=MetricAuditResult(
            skipped="Fairness metrics skipped: no predictions_path configured."
        ),
        shap=ShapAuditResult(
            skipped=(
                "SHAP skipped: eval_path is not configured "
                "(refusing to use the profiled dataset)."
            )
        ),
    )
    markdown = render_bias_identification(result)
    assert "## Bias Identification" in markdown
    assert "### Findings" in markdown
    assert ASSUMPTIONS_HEADING not in markdown
    assert "predictions_path" in markdown
    assert "eval_path" in markdown


def test_skip_reasons_rendered_when_protected_attributes_missing():
    result = BiasIdentificationResult(
        coverage=CoverageResult(
            n_rows=3,
            skipped="Coverage skipped: no protected attributes configured.",
        ),
        metrics=MetricAuditResult(
            skipped="Fairness metrics skipped: no protected attributes configured."
        ),
        shap=ShapAuditResult(
            skipped=(
                "SHAP skipped: eval_path is not configured "
                "(refusing to use the profiled dataset)."
            )
        ),
        skip_reasons=("Coverage skipped: no protected attributes configured.",),
    )
    markdown = render_bias_identification(result)
    assert "no protected attributes configured" in markdown
    assert ASSUMPTIONS_HEADING not in markdown


def test_render_populated_coverage_metrics_and_shap():
    result = _populated_result()
    markdown = render_bias_identification(result)
    visible, _, breakdown = markdown.partition("<details>")
    assert ASSUMPTIONS_HEADING in markdown
    assert "### Findings" in markdown
    assert "underrepresentation flags" in markdown
    assert "`sex` outcomes" in markdown
    assert "`f1` across `sex`" in markdown
    assert "Eval n below 30" in markdown
    assert "Positive class is the `1` label" in markdown
    assert "Question:" not in visible
    assert "With respect to" not in visible
    assert "#### `sex`" in breakdown
    assert "sec:identification" not in markdown
    assert "_Paper:" not in markdown
    assert "Section 6.3" not in markdown
    assert "Underrepresented" in markdown
    assert "Intersectional sparsity" in markdown
    assert "Actual +" in markdown
    assert "Predicted +" in markdown
    assert "Mean abs SHAP" in markdown
    assert "Mean |SHAP|" not in markdown
    assert "Worst-group accuracy" in markdown
    assert "Where outcome metrics come from" in markdown
    assert "Grouped by" in markdown
    assert "Groups in table" in markdown
    assert "f 80.0% (n=8)" in visible
    assert "f 80.0%" not in visible.split("### Representational coverage")[0]


def test_full_report_detail_expands_breakdown():
    markdown = render_bias_identification(_populated_result(), detail="full")
    assert "<details>" not in markdown
    assert "Groups in table" not in markdown
    assert "### Fairness dashboard" in markdown
    assert "Where outcome metrics come from" in markdown
    assert "y_true" in markdown
    assert "Question:" not in markdown


def test_summary_does_not_repeat_outcome_rates_in_findings():
    visible = render_bias_identification(_populated_result()).split("<details>")[0]
    assert "50.0%" in visible
    assert visible.split("### Model outcomes")[0].count("50.0%") == 0


def test_shap_grouping_feature_is_marked_expected():
    result = _populated_result()
    result = BiasIdentificationResult(
        coverage=result.coverage,
        metrics=result.metrics,
        shap=ShapAuditResult(
            n_explained=8,
            global_importance=(ShapFeatureImportance("sex", 0.2),),
            by_group=(
                ShapGroupMean("sex", "f", "sex", 0.3),
                ShapGroupMean("sex", "m", "sex", -0.2),
            ),
            flags=("sex/sex: cross-group mean SHAP gap 0.500 exceeds 0.1",),
        ),
    )
    markdown = render_bias_identification(result)
    assert "expected when the grouping column is a model feature" in markdown
    assert "`f1` across" not in markdown


def test_summary_report_detail_is_the_default():
    markdown = render_bias_identification(_populated_result())
    assert "<details>" in markdown
    assert "Breakdown — full tables" in markdown


def test_normalize_report_detail():
    assert normalize_report_detail("full") == "full"
    assert normalize_report_detail("DETAILED") == "full"
    assert normalize_report_detail("short") == "summary"
    assert normalize_report_detail("nope") == "summary"
    assert normalize_report_detail(None) == "summary"


def test_skip_report_has_no_hidden_breakdown():
    result = BiasIdentificationResult(
        coverage=CoverageResult(
            n_rows=3,
            skipped="Coverage skipped: no protected attributes configured.",
        ),
        metrics=MetricAuditResult(
            skipped="Fairness metrics skipped: no protected attributes configured."
        ),
        shap=ShapAuditResult(
            skipped=(
                "SHAP skipped: eval_path is not configured "
                "(refusing to use the profiled dataset)."
            )
        ),
    )
    markdown = render_bias_identification(result)
    assert "<details>" not in markdown
    assert "no grouping columns" in markdown
