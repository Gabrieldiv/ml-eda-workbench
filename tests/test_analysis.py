"""Unit tests for EDA analysis functions."""

import pandas as pd

from eda_workbench.analysis import basic_info, missing_values, zero_values


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


def test_zero_values_none():
    """Returns 'None' when no characteristics have zero values."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    config = {"id_column": None, "target_column": None}
    result = zero_values(df, config)
    assert "None" in result


def test_zero_values_some():
    """Reports columns with zero values, excluding ID and target."""
    df = pd.DataFrame({"id": [1, 2, 3], "a": [0, 1, 2], "target": [0, 0, 1]})
    config = {"id_column": "id", "target_column": "target"}
    result = zero_values(df, config)
    assert "a - 1 instances out of 3 - 33.3%" in result
    assert "id" not in result
    assert "target" not in result


def test_zero_values_percentage():
    """Calculates percentage correctly."""
    df = pd.DataFrame({"a": [0, 1, 1, 1], "b": [1, 1, 1, 1]})
    config = {"id_column": None, "target_column": None}
    result = zero_values(df, config)
    assert "a - 1 instances out of 4 - 25.0%" in result


def test_zero_values_empty_dataframe():
    """Returns 'None' when DataFrame has no rows."""
    df = pd.DataFrame({"a": [], "b": []})
    config = {"id_column": None, "target_column": None}
    result = zero_values(df, config)
    assert "None" in result


def test_zero_values_ignores_text_columns():
    """Does not break on text columns."""
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [0, 25]})
    config = {"id_column": None, "target_column": None}
    result = zero_values(df, config)
    assert "age - 1 instances out of 2 - 50.0%" in result
    assert "name" not in result


def test_zero_values_empty_config():
    """Works correctly when config has no id or target columns."""
    df = pd.DataFrame({"a": [0, 1, 2], "b": [1, 1, 1]})
    result = zero_values(df, {})
    assert "a - 1 instances out of 3 - 33.3%" in result


def test_zero_values_all_zeros():
    """Reports correctly when all values in a column are zero."""
    df = pd.DataFrame({"a": [0, 0, 0], "b": [1, 2, 3]})
    config = {"id_column": None, "target_column": None}
    result = zero_values(df, config)
    assert "a - 3 instances out of 3 - 100.0%" in result
