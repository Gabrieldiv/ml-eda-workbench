"""Representational coverage on the profiled dataset (pandas only, no model).

Paper: Section 6.3 Dataset and cohort bias auditing
(``sec:dataset-and-cohort-bias-auditing``). Named there as representational
coverage audits that quantify imbalance, including intersectional sparsity.
"""

from __future__ import annotations

import pandas as pd

from eda_workbench.bias.prepare import (
    group_labels,
    imbalance_ratio,
    protected_attributes,
    split_present_attributes,
)
from eda_workbench.bias.types import (
    AttributeCoverage,
    CoverageResult,
    GroupShare,
    IntersectionShare,
)


def compute_coverage(
    frame: pd.DataFrame,
    dataset_config: dict,
    thresholds: dict,
) -> CoverageResult:
    """Summarize group counts, imbalance, and intersectional sparsity.

    Paper: §6.3 (``sec:dataset-and-cohort-bias-auditing``). Group n / share
    are the descriptive statistics from the same section.

    Args:
        frame: The profiled dataset (never an eval table).
        dataset_config: Dataset settings including ``protected_attributes``.
        thresholds: Resolved bias thresholds.

    Returns:
        Coverage findings, or a skipped result when attributes are missing.
    """
    attributes = protected_attributes(dataset_config)
    n_rows = len(frame)
    if not attributes:
        return CoverageResult(
            n_rows=n_rows,
            skipped="Coverage skipped: no protected attributes configured.",
        )

    present, missing = split_present_attributes(frame, attributes)
    notes: list[str] = []
    if missing:
        listed = ", ".join(f"`{name}`" for name in missing)
        notes.append(f"Protected attributes not found in the profiled table: {listed}.")
    if not present:
        return CoverageResult(
            n_rows=n_rows,
            notes=tuple(notes),
            skipped=(
                "Coverage skipped: none of the configured protected "
                "attributes are in the profiled table."
            ),
        )

    min_count = int(thresholds["min_group_count"])
    min_share = float(thresholds["min_group_share"])
    sparse_share = float(thresholds["sparse_intersection_share"])

    attr_results: list[AttributeCoverage] = []
    for name in present:
        labels = group_labels(frame[name])
        n_missing = int(frame[name].isna().sum())
        counts = labels.value_counts()
        groups = []
        for group, count in counts.items():
            share = float(count) / n_rows if n_rows else 0.0
            groups.append(
                GroupShare(
                    attribute=name,
                    group=str(group),
                    count=int(count),
                    share=share,
                    underrepresented=int(count) < min_count or share < min_share,
                )
            )
        attr_results.append(
            AttributeCoverage(
                attribute=name,
                n_missing=n_missing,
                imbalance_ratio=imbalance_ratio(counts),
                groups=tuple(groups),
            )
        )

    intersections: list[IntersectionShare] = []
    if len(present) >= 2:
        combo = pd.DataFrame({name: group_labels(frame[name]) for name in present})
        combo_counts = combo.value_counts()
        for index, count in combo_counts.items():
            values = index if isinstance(index, tuple) else (index,)
            combination = tuple(
                zip(present, (str(value) for value in values), strict=True)
            )
            share = float(count) / n_rows if n_rows else 0.0
            intersections.append(
                IntersectionShare(
                    combination=combination,
                    count=int(count),
                    share=share,
                    sparse=share < sparse_share,
                )
            )

    return CoverageResult(
        n_rows=n_rows,
        attributes=tuple(attr_results),
        intersections=tuple(intersections),
        notes=tuple(notes),
    )
