"""Unit tests for EDA analysis functions."""

import pandas as pd
from eda_workbench.analysis import basic_info, missing_values


def test_missing_values_none():
    """Returns 0 when no missing values exist."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    assert missing_values(df) == 0


def test_missing_values_some():
    """Counts only rows with at least one missing value."""
    df = pd.DataFrame({"a": [1, None, 3], "b": [4.0, 5.0, None]})
    assert missing_values(df) == 2


def test_basic_info_with_id_column():
    """Reports ID column name when detected."""
    df = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    result = basic_info(df, id_column="id")
    assert "ID column detected: id" in result
    assert "Rows      : 2" in result
    assert "Columns   : 2" in result


def test_basic_info_without_id_column():
    """Reports no ID column when none is provided."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    result = basic_info(df, id_column=None)
    assert "No ID column detected." in result