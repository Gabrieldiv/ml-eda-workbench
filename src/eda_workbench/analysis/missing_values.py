import pandas as pd

from eda_workbench.analysis.types import MissingValuesResult


def missing_values(df: pd.DataFrame) -> MissingValuesResult:
    """Count how many instances have at least one missing value.

    Args:
        df: DataFrame with the dataset data.

    Returns:
        Structured result with the count of instances with at least one
        missing value, and the total number of instances.
    """
    rows_with_missing = int(df.isna().any(axis=1).sum())
    return MissingValuesResult(
        rows_with_missing=rows_with_missing,
        total_rows=len(df),
    )
