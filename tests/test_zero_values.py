"""Unit tests for zero_values."""

import pandas as pd

from eda_workbench.analysis import zero_values


def test_zero_values_none():
    """Returns a 'no zeros' message when no characteristics have zero values."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    config = {"id_column": None, "target_column": None}
    result = zero_values(df, config)
    assert "**No characteristics with zero values**" in result


def test_zero_values_some():
    """Reports columns with zero values, excluding ID and target."""
    df = pd.DataFrame({"id": [1, 2, 3], "a": [0, 1, 2], "target": [0, 0, 1]})
    config = {"id_column": "id", "target_column": "target"}
    result = zero_values(df, config)
    assert "1 instances out of 3" in result
    assert "33.3%" in result
    assert "| a " in result
    assert "| id " not in result
    assert "| target " not in result


def test_zero_values_percentage():
    """Calculates percentage correctly."""
    df = pd.DataFrame({"a": [0, 1, 1, 1], "b": [1, 1, 1, 1]})
    config = {"id_column": None, "target_column": None}
    result = zero_values(df, config)
    assert "1 instances out of 4" in result
    assert "25.0%" in result


def test_zero_values_empty_dataframe():
    """Returns a 'no zeros' message when DataFrame has no rows."""
    df = pd.DataFrame({"a": [], "b": []})
    config = {"id_column": None, "target_column": None}
    result = zero_values(df, config)
    assert "**No characteristics with zero values**" in result


def test_zero_values_ignores_text_columns():
    """Does not break on text columns."""
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [0, 25]})
    config = {"id_column": None, "target_column": None}
    result = zero_values(df, config)
    assert "1 instances out of 2" in result
    assert "50.0%" in result
    assert "| age " in result
    assert "| name " not in result


def test_zero_values_empty_config():
    """Works correctly when config has no id or target columns."""
    df = pd.DataFrame({"a": [0, 1, 2], "b": [1, 1, 1]})
    result = zero_values(df, {})
    assert "1 instances out of 3" in result
    assert "33.3%" in result


def test_zero_values_all_zeros():
    """Reports correctly when all values in a column are zero."""
    df = pd.DataFrame({"a": [0, 0, 0], "b": [1, 2, 3]})
    config = {"id_column": None, "target_column": None}
    result = zero_values(df, config)
    assert "3 instances out of 3" in result
    assert "100.0%" in result
