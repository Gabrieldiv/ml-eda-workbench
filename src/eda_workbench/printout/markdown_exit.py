"""Markdown renderers for structured analysis results."""

from eda_workbench.analysis.types import MissingValuesResult


def render_missing_values(result: MissingValuesResult) -> str:
    """Format a MissingValuesResult as a Markdown section.

    Args:
        result: Structured result from missing_values().

    Returns:
        Markdown-formatted string with a "Missing Values" heading and the
        instance count with missing data, out of the total.
    """
    return (
        "## Missing Values\n"
        f"- **Instances with missing values**: {result.rows_with_missing} "
        f"out of {result.total_rows}"
    )
