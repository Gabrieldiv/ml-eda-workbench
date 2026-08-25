"""Markdown renderer for bias-identification findings."""

from __future__ import annotations

import math
from collections import defaultdict

from eda_workbench.bias.prepare import resolve_thresholds
from eda_workbench.bias.types import (
    AttributeMetricSummary,
    BiasIdentificationResult,
    CoverageResult,
    GroupMetrics,
    MetricAuditResult,
    ShapAuditResult,
)

ASSUMPTIONS_HEADING = "### Assumptions"

ASSUMPTIONS_BODY = """\
The fairness dashboard and/or SHAP attributions below use a user-supplied \
evaluation bundle. The workbench cannot verify these assumptions:

- `eval_path` / `predictions_path` are held-out (or otherwise the set intended \
for audit), not the training rows.
- `y_pred` / `y_score` were produced by the supplied model (or the system \
being audited) on those same eval rows.
- Protected-attribute labels in the eval bundle are the ones that should be \
used for grouping.
- When both files are given, row alignment between `eval_path` and \
`predictions_path` (by id) is correct.

Representational coverage describes the **profiled table's composition** \
only; it is not a claim about generalization of any model.\
"""

BREAKDOWN_SUMMARY = (
    "Breakdown — full tables, assumptions, and where outcome metrics come from"
)
_SUMMARY_ALIASES = {"summary", "short", "compact"}
_FULL_ALIASES = {"full", "detailed", "long"}


def normalize_report_detail(value: object) -> str:
    """Return ``summary`` or ``full`` from a YAML / caller value."""
    text = str(value or "summary").strip().lower()
    if text in _FULL_ALIASES:
        return "full"
    if text in _SUMMARY_ALIASES or not text:
        return "summary"
    return "summary"


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_share(share: float) -> str:
    return f"{share:.1%}"


def _code_join(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def _md_cell(value: str) -> str:
    """Keep Markdown tables valid: pipes inside cells break column parsing."""
    return value.replace("|", "/")


def _md_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    if not rows:
        return []
    headers = tuple(_md_cell(header) for header in headers)
    rows = [tuple(_md_cell(cell) for cell in row) for row in rows]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    header_line = (
        "| "
        + " | ".join(
            header.ljust(width) for header, width in zip(headers, widths, strict=True)
        )
        + " |"
    )
    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    lines = [header_line, separator]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                cell.ljust(width) for cell, width in zip(row, widths, strict=True)
            )
            + " |"
        )
    return lines


def _md_details(summary: str, body: list[str]) -> list[str]:
    return ["<details>", f"<summary>{summary}</summary>", "", *body, "", "</details>"]


def _render_notes(notes: tuple[str, ...]) -> list[str]:
    return [f"- {note}" for note in notes]


def _grouping_columns(
    result: BiasIdentificationResult, configured: list[str] | None
) -> list[str]:
    if configured is not None:
        return [str(name) for name in configured if str(name).strip()]
    names: list[str] = []
    for attribute in result.coverage.attributes:
        if attribute.attribute not in names:
            names.append(attribute.attribute)
    for row in result.metrics.by_group:
        if row.attribute not in names:
            names.append(row.attribute)
    return names


def _group_metrics(metrics: MetricAuditResult, attribute: str) -> list[GroupMetrics]:
    return [row for row in metrics.by_group if row.attribute == attribute]


def _minmax(
    rows: list[GroupMetrics], field: str
) -> tuple[GroupMetrics, GroupMetrics] | None:
    valued = [row for row in rows if getattr(row, field) is not None]
    if len(valued) < 2:
        return None
    return (
        min(valued, key=lambda row: getattr(row, field)),
        max(valued, key=lambda row: getattr(row, field)),
    )


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "n/a"
    return f"{value:.1%}"


def _range_text(rows: list[GroupMetrics], field: str) -> str:
    pair = _minmax(rows, field)
    if pair is None:
        return "—"
    low, high = pair
    fmt = _fmt_rate if field in _RATE_FIELDS else _fmt_float
    return (
        f"{fmt(getattr(low, field))} (`{low.group}`) – "
        f"{fmt(getattr(high, field))} (`{high.group}`)"
    )


_RATE_FIELDS = {
    "actual_positive_rate",
    "selection_rate",
    "true_positive_rate",
    "false_positive_rate",
    "precision",
    "accuracy",
}


def _shap_gaps(
    shap: ShapAuditResult,
) -> list[tuple[str, str, str, str, float, float, float]]:
    buckets: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for item in shap.by_group:
        buckets[(item.attribute, item.feature)][item.group] = item.mean_shap
    gaps: list[tuple[str, str, str, str, float, float, float]] = []
    for (attribute, feature), means in buckets.items():
        if len(means) < 2:
            continue
        low_group = min(means, key=lambda group: means[group])
        high_group = max(means, key=lambda group: means[group])
        gaps.append(
            (
                attribute,
                feature,
                low_group,
                high_group,
                means[low_group],
                means[high_group],
                means[high_group] - means[low_group],
            )
        )
    gaps.sort(key=lambda item: item[6], reverse=True)
    return gaps


def _is_grouping_feature(attribute: str, feature: str) -> bool:
    return attribute.strip().lower() == feature.strip().lower()


def _small_eval_groups(groups: list[GroupMetrics], min_n: int) -> list[GroupMetrics]:
    return [row for row in groups if row.count < min_n]


def _flagged_check_names(
    summary: AttributeMetricSummary, thresholds: dict
) -> list[str]:
    rate_thr = float(thresholds["rate_difference_threshold"])
    di_thr = float(thresholds["disparate_impact_threshold"])
    names: list[str] = []
    if (
        summary.demographic_parity_difference is not None
        and abs(summary.demographic_parity_difference) > rate_thr
    ):
        names.append("predicted-positive rate")
    if (
        summary.demographic_parity_ratio is not None
        and summary.demographic_parity_ratio < di_thr
    ):
        names.append("predicted-positive ratio")
    if (
        summary.equal_opportunity_difference is not None
        and abs(summary.equal_opportunity_difference) > rate_thr
    ):
        names.append("TPR")
    if (
        summary.equalized_odds_difference is not None
        and abs(summary.equalized_odds_difference) > rate_thr
    ):
        names.append("equalized odds")
    if summary.accuracy_gap is not None and summary.accuracy_gap > rate_thr:
        names.append("accuracy")
    return names


def _render_intro(
    result: BiasIdentificationResult,
    grouping_columns: list[str],
    target_column: str | None,
) -> list[str]:
    if grouping_columns:
        grouped = f"Grouped by {_code_join(grouping_columns)} (`protected_attributes`)."
    else:
        grouped = (
            "No `protected_attributes` were configured, so there are no "
            "grouping columns to audit."
        )
    if result.ran_metrics:
        outcome = f"`{target_column}`" if target_column else "`y_true`"
        positive = (
            f" Positive class is the `1` label on {outcome} (`y_true` / `y_pred`)."
        )
    else:
        positive = ""
    return [
        f"{grouped}{positive} Flags are threshold crossings for a grouping "
        "column — not a verdict that the dataset or model “is biased”.",
    ]


def _render_coverage_findings(coverage: CoverageResult, thresholds: dict) -> list[str]:
    if coverage.skipped:
        return [f"- **Table:** {coverage.skipped}"]

    min_n = int(thresholds["min_group_count"])
    min_share = float(thresholds["min_group_share"])
    under = [
        f"`{group.attribute}`=`{group.group}` (n={group.count})"
        for attribute in coverage.attributes
        for group in attribute.groups
        if group.underrepresented
    ]
    sparse_n = sum(1 for item in coverage.intersections if item.sparse)
    if under:
        extra = f"; {sparse_n} sparse intersection(s)" if sparse_n else ""
        return [f"- **Table:** underrepresentation flags: {', '.join(under)}{extra}."]
    if sparse_n:
        return [
            f"- **Table:** {sparse_n} intersectional combination(s) below the "
            "share cutoff (see table)."
        ]
    return [
        f"- **Table:** no underrepresentation flags "
        f"(n<{min_n} or share<{_fmt_share(min_share)})."
    ]


def _render_metric_findings(
    result: BiasIdentificationResult, thresholds: dict
) -> list[str]:
    metrics = result.metrics
    if not result.ran_metrics:
        reason = metrics.skipped or "no predictions table loaded."
        return [f"- **Outcomes:** not computed ({reason})"]

    min_n = int(thresholds["min_group_count"])
    lines: list[str] = []
    for summary in metrics.summaries:
        groups = _group_metrics(metrics, summary.attribute)
        checks = _flagged_check_names(summary, thresholds)
        small = _small_eval_groups(groups, min_n)
        small_note = ""
        if small:
            listed = ", ".join(f"`{row.group}` (n={row.count})" for row in small)
            small_note = f" Eval n below {min_n}: {listed}."
        if checks:
            listed = ", ".join(checks)
            lines.append(
                f"- **`{summary.attribute}` outcomes:** flagged ({listed}).{small_note}"
            )
        else:
            lines.append(
                f"- **`{summary.attribute}` outcomes:** no rate gaps above "
                f"thresholds.{small_note} Not a proof of fairness."
            )
    lines.extend(_render_notes(metrics.notes))
    return lines


def _render_shap_findings(
    result: BiasIdentificationResult, thresholds: dict
) -> list[str]:
    shap = result.shap
    if not result.ran_shap:
        reason = shap.skipped or "eval table or model was not loaded."
        return [f"- **SHAP:** not computed ({reason})"]

    gap_thr = float(thresholds["shap_attribution_gap_threshold"])
    flagged = [item for item in _shap_gaps(shap) if item[6] > gap_thr]

    expected = [item for item in flagged if _is_grouping_feature(item[0], item[1])]
    other = [item for item in flagged if not _is_grouping_feature(item[0], item[1])]
    lines: list[str] = []
    if other:
        bits = [
            f"`{feature}` across `{attribute}`" for attribute, feature, *_rest in other
        ]
        lines.append(
            f"- **SHAP:** attribution gaps above {gap_thr:g}: {', '.join(bits)}."
        )
    if expected:
        bits = [f"`{feature}`" for _attr, feature, *_rest in expected]
        lines.append(
            "- **SHAP:** "
            + ", ".join(bits)
            + " differ across their own groups (expected when the grouping "
            "column is a model feature)."
        )
    if not flagged:
        if shap.flags:
            lines.extend(f"- **SHAP:** {flag}" for flag in shap.flags)
        else:
            lines.append(
                f"- **SHAP:** no cross-group mean SHAP gaps above {gap_thr:g}."
            )
    lines.extend(_render_notes(shap.notes))
    return lines


def _render_findings(result: BiasIdentificationResult, thresholds: dict) -> list[str]:
    lines = ["### Findings", ""]
    lines.extend(_render_coverage_findings(result.coverage, thresholds))
    lines.extend(_render_metric_findings(result, thresholds))
    lines.extend(_render_shap_findings(result, thresholds))
    return lines


def _render_coverage_compact(coverage: CoverageResult) -> list[str]:
    lines = [
        "### Representational coverage",
        "",
        f"{coverage.n_rows} profiled rows. Shares are how common each group "
        "is in this table, not model quality.",
        "",
    ]
    if coverage.skipped:
        lines.append(f"- {coverage.skipped}")
        return lines

    rows = []
    for attribute in coverage.attributes:
        groups = ", ".join(
            f"{group.group} {_fmt_share(group.share)} (n={group.count})"
            for group in attribute.groups
        )
        under = (
            ", ".join(
                group.group for group in attribute.groups if group.underrepresented
            )
            or "none"
        )
        rows.append(
            (
                f"`{attribute.attribute}`",
                groups,
                _fmt_float(attribute.imbalance_ratio, digits=2),
                under,
            )
        )
    lines.extend(
        _md_table(("Column", "Groups in table", "Imbalance", "Underrepresented"), rows)
    )
    sparse_n = sum(1 for item in coverage.intersections if item.sparse)
    if sparse_n:
        lines.append("")
        lines.append(f"{sparse_n} sparse intersection(s) — see breakdown.")
    return lines


def _render_metrics_compact(metrics: MetricAuditResult) -> list[str]:
    lines = [
        "### Model outcomes",
        "",
        f"{metrics.n_rows} eval rows. **Actual +** is label prevalence; "
        "**Predicted +** is the model's positive rate. Compare the two to "
        "see whether a prediction gap tracks the labels.",
        "",
    ]
    rows = []
    for summary in metrics.summaries:
        groups = _group_metrics(metrics, summary.attribute)
        rows.append(
            (
                f"`{summary.attribute}`",
                _range_text(groups, "actual_positive_rate"),
                _range_text(groups, "selection_rate"),
                _range_text(groups, "true_positive_rate"),
                _range_text(groups, "accuracy"),
            )
        )
    lines.extend(
        _md_table(
            (
                "Grouped by",
                "Actual +",
                "Predicted +",
                "TPR",
                "Accuracy",
            ),
            rows,
        )
    )
    return lines


def _render_shap_compact(shap: ShapAuditResult) -> list[str]:
    lines = [
        "### Feature attributions (SHAP)",
        "",
        f"{shap.n_explained} eval rows explained. Higher mean abs SHAP means "
        "the feature moved the model's output more.",
        "",
    ]
    top = shap.global_importance[:5]
    if top:
        rows = [(item.feature, _fmt_float(item.mean_abs_shap)) for item in top]
        lines.extend(_md_table(("Feature", "Mean abs SHAP"), rows))
    return lines


def _render_coverage(coverage: CoverageResult) -> list[str]:
    lines = ["### Representational coverage", ""]
    if coverage.skipped:
        lines.append(f"- {coverage.skipped}")
        lines.extend(_render_notes(coverage.notes))
        return lines

    lines.append(
        f"- **Profiled rows**: {coverage.n_rows} "
        "(table composition only; not a model-generalization claim)"
    )
    lines.extend(_render_notes(coverage.notes))

    for coverage_attr in coverage.attributes:
        lines.extend(["", f"#### `{coverage_attr.attribute}`", ""])
        lines.append(f"- **Missing values**: {coverage_attr.n_missing}")
        lines.append(
            "- **Imbalance ratio** (largest/smallest group): "
            f"{_fmt_float(coverage_attr.imbalance_ratio, digits=2)}"
        )
        rows = [
            (
                group.group,
                str(group.count),
                _fmt_share(group.share),
                "yes" if group.underrepresented else "no",
            )
            for group in coverage_attr.groups
        ]
        lines.append("")
        lines.extend(_md_table(("Group", "n", "Share", "Underrepresented"), rows))

    if coverage.intersections:
        lines.extend(["", "#### Intersectional sparsity", ""])
        rows = [
            (
                ", ".join(f"{name}={value}" for name, value in item.combination),
                str(item.count),
                _fmt_share(item.share),
                "yes" if item.sparse else "no",
            )
            for item in coverage.intersections
        ]
        lines.extend(_md_table(("Combination", "n", "Share", "Sparse"), rows))
    return lines


def _render_metrics(metrics: MetricAuditResult) -> list[str]:
    lines = ["### Fairness dashboard", ""]
    if metrics.skipped:
        lines.append(f"- {metrics.skipped}")
        lines.extend(_render_notes(metrics.notes))
        return lines

    target_kind = "binary" if metrics.binary_target else "non-binary"
    lines.append(f"- **Eval prediction rows**: {metrics.n_rows} ({target_kind} target)")
    if not metrics.binary_target:
        lines.append(
            "- Binary independence/separation/sufficiency metrics omitted; "
            "group-aggregate accuracy is reported."
        )
    lines.extend(_render_notes(metrics.notes))

    if metrics.by_group:
        lines.append("")
        rows = [
            (
                row.attribute,
                row.group,
                str(row.count),
                _fmt_rate(row.actual_positive_rate),
                _fmt_rate(row.selection_rate),
                _fmt_rate(row.true_positive_rate),
                _fmt_rate(row.false_positive_rate),
                _fmt_rate(row.precision),
                _fmt_rate(row.accuracy),
                _fmt_float(row.roc_auc),
            )
            for row in metrics.by_group
        ]
        lines.extend(
            _md_table(
                (
                    "Attribute",
                    "Group",
                    "n",
                    "Actual +",
                    "Predicted +",
                    "TPR",
                    "FPR",
                    "Precision",
                    "Accuracy",
                    "AUC",
                ),
                rows,
            )
        )

    for summary in metrics.summaries:
        lines.extend(["", f"#### `{summary.attribute}` summaries", ""])
        lines.append(
            f"- **Demographic parity difference**: "
            f"{_fmt_float(summary.demographic_parity_difference)}"
        )
        lines.append(
            f"- **Demographic parity ratio**: "
            f"{_fmt_float(summary.demographic_parity_ratio)}"
        )
        lines.append(
            f"- **Equal opportunity difference**: "
            f"{_fmt_float(summary.equal_opportunity_difference)}"
        )
        lines.append(
            f"- **Equalized odds difference**: "
            f"{_fmt_float(summary.equalized_odds_difference)}"
        )
        lines.append(
            f"- **Worst-group accuracy**: {_fmt_float(summary.worst_group_accuracy)}"
        )
        lines.append(f"- **Accuracy gap**: {_fmt_float(summary.accuracy_gap)}")
        if summary.flags:
            lines.append("- **Flags**:")
            lines.extend(f"  - {flag}" for flag in summary.flags)
    return lines


def _render_shap(result: ShapAuditResult) -> list[str]:
    lines = ["### SHAP", ""]
    if result.skipped:
        lines.append(f"- {result.skipped}")
        lines.extend(_render_notes(result.notes))
        return lines

    lines.append(f"- **Eval rows explained**: {result.n_explained}")
    lines.extend(_render_notes(result.notes))

    if result.global_importance:
        lines.extend(["", "#### Mean abs SHAP (global)", ""])
        top = result.global_importance[:10]
        rows = [(item.feature, _fmt_float(item.mean_abs_shap)) for item in top]
        lines.extend(_md_table(("Feature", "Mean abs SHAP"), rows))

    if result.by_group:
        lines.extend(["", "#### Mean SHAP by group", ""])
        rows = [
            (
                item.attribute,
                item.group,
                item.feature,
                _fmt_float(item.mean_shap),
            )
            for item in result.by_group
        ]
        lines.extend(_md_table(("Attribute", "Group", "Feature", "Mean SHAP"), rows))

    if result.flags:
        lines.extend(["", "- **Flags**:"])
        for flag in result.flags:
            prefix = flag.split(":", 1)[0]
            note = ""
            if "/" in prefix:
                attribute, feature = prefix.split("/", 1)
                if _is_grouping_feature(attribute, feature):
                    note = " (expected: grouping column is a model feature)"
            lines.append(f"  - {flag}{note}")
    return lines


def _render_glossary() -> list[str]:
    return [
        "### Where outcome metrics come from",
        "",
        "The dashboard does **not** read precision / accuracy / AUC from the "
        "profiled CSV. When `predictions_path` is set, fairlearn `MetricFrame` "
        "and sklearn compute them on that file, grouped by each "
        "`protected_attributes` column:",
        "",
        "- `y_true` (or the dataset `target_column`): actual labels",
        "- `y_pred`: predicted labels already written to the predictions file",
        "- `y_score` (optional): scores or probabilities; used only for AUC",
        "",
        "| Name | Question | Source |",
        "|------|----------|--------|",
        "| Actual + | What share of this group is actually positive "
        "(`y_true`)? | fairlearn `selection_rate` on `y_true` |",
        "| Predicted + | What share of this group got a positive `y_pred`? "
        "| fairlearn `selection_rate` |",
        "| TPR | Of actual positives, how many were predicted positive? "
        "| fairlearn `true_positive_rate` |",
        "| FPR | Of actual negatives, how many were predicted positive? "
        "| fairlearn `false_positive_rate` |",
        "| Precision | Of predicted positives, how many were actually "
        "positive? | sklearn `precision_score` |",
        "| Accuracy | How often does `y_pred` match `y_true`? "
        "| sklearn `accuracy_score` |",
        "| AUC | How well do `y_score` values rank the two classes? "
        "| sklearn `roc_auc_score` (needs `y_score`) |",
        "",
        "Demographic parity, equal opportunity, and equalized odds are "
        "cross-group summaries of those rates (fairlearn). SHAP values come "
        "from `shap.PermutationExplainer` on `eval_path` + `model_path` only.",
    ]


def _render_assumptions() -> list[str]:
    return [ASSUMPTIONS_HEADING, "", ASSUMPTIONS_BODY]


def _has_breakdown(result: BiasIdentificationResult) -> bool:
    coverage_tables = result.coverage.skipped is None and bool(
        result.coverage.attributes
    )
    return coverage_tables or result.ran_metrics or result.ran_shap


def _render_full_sections(
    result: BiasIdentificationResult, *, include_assumptions: bool
) -> list[str]:
    lines: list[str] = []
    if include_assumptions and (result.ran_metrics or result.ran_shap):
        lines.extend(_render_assumptions())
        lines.append("")
    lines.extend(_render_coverage(result.coverage))
    lines.append("")
    lines.extend(_render_metrics(result.metrics))
    lines.append("")
    lines.extend(_render_shap(result.shap))
    lines.append("")
    lines.extend(_render_glossary())
    return lines


def render_bias_identification(
    result: BiasIdentificationResult,
    *,
    detail: object = "summary",
    target_column: str | None = None,
    grouping_columns: list[str] | tuple[str, ...] | None = None,
    thresholds: dict | None = None,
) -> str:
    """Render coverage, optional fairness dashboard, and optional SHAP.

    ``detail='summary'`` (default) keeps compact tables visible and puts the
    full breakdown in a collapsible ``<details>`` block. ``detail='full'``
    expands everything.
    """
    mode = normalize_report_detail(detail)
    resolved = resolve_thresholds(thresholds)
    grouping = _grouping_columns(
        result, list(grouping_columns) if grouping_columns is not None else None
    )

    lines = ["## Bias Identification", ""]
    lines.extend(_render_intro(result, grouping, target_column))
    lines.append("")
    lines.extend(_render_findings(result, resolved))
    lines.append("")

    if mode == "full":
        lines.extend(_render_full_sections(result, include_assumptions=True))
        return "\n".join(lines)

    if result.coverage.skipped is None and result.coverage.attributes:
        lines.extend(_render_coverage_compact(result.coverage))
        lines.append("")
    if result.ran_metrics and result.metrics.summaries:
        lines.extend(_render_metrics_compact(result.metrics))
        lines.append("")
    if result.ran_shap:
        lines.extend(_render_shap_compact(result.shap))
        lines.append("")

    if _has_breakdown(result):
        lines.extend(
            _md_details(
                BREAKDOWN_SUMMARY,
                _render_full_sections(result, include_assumptions=True),
            )
        )
    return "\n".join(lines)
