"""Unit tests for outliers_instances."""

import pandas as pd

from eda_workbench.analysis import missing_values


def test_missing_values_none():
    """Returns 0 rows with missing values when none exist."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = missing_values(df)
    assert result.rows_with_missing == 0
    assert result.total_rows == 3


def test_missing_values_some():
    """Counts only rows with at least one missing value."""
    df = pd.DataFrame({"a": [1, None, 3], "b": [4.0, 5.0, None]})
    result = missing_values(df)
    assert result.rows_with_missing == 2
    assert result.total_rows == 3
