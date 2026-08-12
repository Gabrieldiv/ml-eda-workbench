"""Unit tests for outliers_instances."""

import numpy as np
import pandas as pd

from eda_workbench.analysis import outliers_instances


def test_finds_outlier_with_id():
    """Detects an outlier and reports it using the ID column."""
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "a": [1, 2, 3, 4, 100]})
    config = {"id_column": "id", "target_column": None}
    result = outliers_instances(df, config, 1.5)
    assert "5: 100.00" in result
    assert "| a" in result


def test_no_outliers_found():
    """Reports zero outliers and a dash when no value exceeds the whisker range."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    config = {"id_column": None, "target_column": None}
    result = outliers_instances(df, config, 1.5)
    assert "| a" in result
    assert "0" in result
    assert "—" in result


def test_uses_row_index_without_id_column():
    """Falls back to row index when there is no ID column configured."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100]})
    config = {"id_column": None, "target_column": None}
    result = outliers_instances(df, config, 1.5)
    assert "4: 100.00" in result


def test_empty_dataset():
    """Reports an empty dataset without crashing."""
    df = pd.DataFrame()
    config = {"id_column": None, "target_column": None}
    result = outliers_instances(df, config, 1.5)
    assert "**Empty dataset**" in result


def test_only_id_and_target_columns():
    """Reports that nothing is left to analyze when only ID and target remain."""
    df = pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]})
    config = {"id_column": "id", "target_column": "target"}
    result = outliers_instances(df, config, 1.5)
    assert "no analyzable data remains" in result


def test_only_non_numeric_columns():
    """Reports skipped columns and that no numeric characteristics were found."""
    df = pd.DataFrame({"nome": ["Ana", "Bruno", "Carlos"]})
    config = {"id_column": None, "target_column": None}
    result = outliers_instances(df, config, 1.5)
    assert "**Skipped non-numeric characteristics**: nome" in result
    assert "**No numeric characteristics found**" in result


def test_mixed_numeric_and_non_numeric_columns():
    """Skips non-numeric columns but still processes numeric ones."""
    df = pd.DataFrame({"nome": ["Ana", "Bruno"], "idade": [10, 20]})
    config = {"id_column": None, "target_column": None}
    result = outliers_instances(df, config, 1.5)
    assert "**Skipped non-numeric characteristics**: nome" in result
    assert "| idade" in result


def test_column_with_no_valid_data():
    """Reports a column with no valid data separately, without breaking others."""
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1, 2, 3]})
    config = {"id_column": None, "target_column": None}
    result = outliers_instances(df, config, 1.5)
    assert "**Characteristics with no valid data**: a" in result
    assert "| b" in result
    assert "| a" not in result


def test_all_columns_with_no_valid_data():
    """Reports that no numeric characteristic has valid data when all are empty."""
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    config = {"id_column": None, "target_column": None}
    result = outliers_instances(df, config, 1.5)
    assert "**No numeric characteristics with valid data**" in result


def test_negative_whisker_multiplier_rejected():
    """Rejects a negative whisker_multiplier instead of computing invalid bounds."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    config = {"id_column": None, "target_column": None}
    result = outliers_instances(df, config, -1.5)
    assert "must be >= 0" in result


def test_zero_whisker_multiplier_flags_extremes():
    """A whisker_multiplier of zero flags min and max as outliers."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    config = {"id_column": None, "target_column": None}
    result = outliers_instances(df, config, 0)
    assert "2" in result  # 2 outliers: min and max


def test_missing_id_shows_placeholder_not_nan():
    """Shows a clear placeholder instead of the literal 'nan' when the ID is missing."""
    df = pd.DataFrame({"id": [1, 2, 3, 4, np.nan], "a": [1, 2, 3, 4, 100]})
    config = {"id_column": "id", "target_column": None}
    result = outliers_instances(df, config, 1.5)
    assert "missing ID" in result
    assert "nan:" not in result


def test_single_valid_value_in_column():
    """Computes stats without crashing when only one value is valid."""
    df = pd.DataFrame({"a": [np.nan, np.nan, 42, np.nan]})
    config = {"id_column": None, "target_column": None}
    result = outliers_instances(df, config, 1.5)
    assert "42.00" in result
    assert "0" in result


def test_ignores_id_and_target_columns():
    """Never treats the ID or target columns as characteristics to analyze."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
            "a": [1, 2, 3, 4, 5],
        }
    )
    config = {"id_column": "id", "target_column": "target"}
    result = outliers_instances(df, config, 1.5)
    assert "| id" not in result
    assert "| target" not in result
    assert "| a" in result


def test_truncates_long_outlier_lists():
    """Truncates the outlier list to the first 10 entries when there are more,
    while keeping the total count accurate."""
    df = pd.DataFrame(
        {
            "id": range(115),
            "a": list(range(1, 11)) * 10 + list(range(1000, 1015)),
        }
    )
    config = {"id_column": "id", "target_column": None}
    result = outliers_instances(df, config, 1.5)
    assert "15" in result
    assert "... and 5 more" in result


def test_zero_iqr_reports_frequency_distribution_instead_of_outliers():
    """When IQR is zero, reports value frequency distribution instead of
    treating every non-mode value as an outlier."""
    df = pd.DataFrame(
        {
            "id": range(100),
            "a": [0] * 80 + [1] * 15 + [2] * 5,
        }
    )
    config = {"id_column": "id", "target_column": None}
    result = outliers_instances(df, config, 1.5)
    assert "N/A (zero IQR)" in result
    assert "0: 80 (80.0%)" in result
    assert "1: 15 (15.0%)" in result
    assert "2: 5 (5.0%)" in result


def test_zero_iqr_frequency_distribution_truncated_when_many_distinct_values():
    """Truncates the frequency distribution list when a zero-IQR characteristic
    has more than 10 distinct values, while keeping the format consistent."""
    df = pd.DataFrame(
        {
            "id": range(100),
            "a": [0] * 80 + list(range(1, 21)),
        }
    )
    config = {"id_column": "id", "target_column": None}
    result = outliers_instances(df, config, 1.5)
    assert "N/A (zero IQR)" in result
    assert "0: 80 (80.0%)" in result
    assert "... and 11 more" in result
