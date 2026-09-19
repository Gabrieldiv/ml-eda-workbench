"""Structured result types for EDA analysis functions.

Each analysis function returns one of these dataclasses instead of a
pre-formatted string, so the same computed result can be rendered into
multiple output formats (Markdown, JSON, PDF) without recomputing
anything. Mirrors the same separation already used in ``bias/types.py``.
"""

from dataclasses import dataclass


@dataclass
class MissingValuesResult:
    """Result of counting instances with at least one missing value.

    Attributes:
        rows_with_missing: Number of instances (rows) with at least one
            missing (NaN) value in any column.
        total_rows: Total number of instances in the dataset.
    """

    rows_with_missing: int
    total_rows: int
