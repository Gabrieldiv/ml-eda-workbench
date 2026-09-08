"""Unit tests for class_distribution."""

import numpy as np
import pandas as pd

from eda_workbench.analysis import class_distribution


def test_no_label_columns_configured():
    """Reports that no label columns were configured."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = class_distribution(df, {})
    assert "**No label columns configured**" in result


def test_duplicate_label_columns_refuses_to_run():
    """Refuses to run when the same column is configured more than once,
    rather than silently processing it twice or picking one."""
    df = pd.DataFrame({"color": ["a", "b"]})
    config = {"label_columns": ["color", "color"]}
    result = class_distribution(df, config)
    assert "**Duplicate label column(s) configured**: color" in result
    assert "### color" not in result


def test_all_configured_columns_missing_from_dataset():
    """Reports the missing column and does not attempt further processing."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    config = {"label_columns": ["nao_existe"]}
    result = class_distribution(df, config)
    assert "**Configured label column(s) not found in dataset**: nao_existe" in result


def test_some_configured_columns_missing_continues_with_present_ones():
    """Reports the missing column but still processes the ones that exist."""
    df = pd.DataFrame({"color": ["a", "b"]})
    config = {"label_columns": ["color", "nao_existe"]}
    result = class_distribution(df, config)
    assert "**Configured label column(s) not found in dataset**: nao_existe" in result
    assert "### color" in result
    assert "**a**: 1 out of 2 (50.0%)" in result


def test_empty_dataset():
    """Reports an empty dataset without crashing."""
    df = pd.DataFrame({"color": pd.Series([], dtype=object)})
    config = {"label_columns": ["color"]}
    result = class_distribution(df, config)
    assert "**Empty dataset**" in result


def test_single_column_distribution_sorted_by_frequency():
    """Reports frequency and percentage per distinct value, most frequent first."""
    df = pd.DataFrame({"specie": ["a", "a", "b"]})
    config = {"label_columns": ["specie"]}
    result = class_distribution(df, config)
    assert "### specie" in result
    assert "**a**: 2 out of 3 (66.7%)" in result
    assert "**b**: 1 out of 3 (33.3%)" in result


def test_accepts_scalar_string_as_single_column():
    """label_columns configured as a plain string (not a list) is normalized
    to a single-item list, not iterated character by character."""
    df = pd.DataFrame({"specie": ["a", "b", "a"]})
    config = {"label_columns": "specie"}
    result = class_distribution(df, config)
    assert "### specie" in result
    assert result.count("###") == 1


def test_missing_values_kept_distinct_from_genuine_none_label():
    """NaN is counted separately from a real 'None' category, and displayed
    as '<missing>' — merging them before counting would overstate the real
    category and hide missingness entirely."""
    df = pd.DataFrame({"color": ["None", "None", "yellow", np.nan, np.nan, np.nan]})
    config = {"label_columns": ["color"]}
    result = class_distribution(df, config)
    assert "**<missing>**: 3 out of 6 (50.0%)" in result
    assert "**None**: 2 out of 6 (33.3%)" in result
    assert "**yellow**: 1 out of 6 (16.7%)" in result


def test_high_cardinality_column_is_truncated_with_note():
    """A column with more distinct values than the threshold shows only the
    most frequent ones, with a note explaining the truncation."""
    df = pd.DataFrame({"id_like": [f"id_{i}" for i in range(30)]})
    config = {"label_columns": ["id_like"]}
    result = class_distribution(df, config, high_cardinality_threshold=5)
    assert "**High cardinality (30 distinct labels) — showing top 5**" in result
    assert result.count("out of 30") == 5


def test_multiple_columns_report_most_common_combination():
    """With more than one configured column, also reports the single most
    frequent combination of values across all of them."""
    df = pd.DataFrame(
        {
            "color": ["yellow", "yellow", "blue", "yellow", "yellow"],
            "size": ["short", "short", "tall", "tall", "short"],
        }
    )
    config = {"label_columns": ["color", "size"]}
    result = class_distribution(df, config)
    assert "### color" in result
    assert "### size" in result
    assert (
        "**Most common multilabel combination**: color = yellow, size = short — "
        "3 out of 5 (60.0%)" in result
    )


def test_combination_includes_missing_value_as_its_own_category():
    """A row with a missing value in one of the combined columns still
    counts toward the combination, showing '<missing>' for that column."""
    df = pd.DataFrame(
        {
            "color": ["yellow", "yellow", "blue", "yellow", np.nan],
            "size": ["short", "short", "tall", "tall", "short"],
        }
    )
    config = {"label_columns": ["color", "size"]}
    result = class_distribution(df, config)
    assert "**<missing>**: 1 out of 5 (20.0%)" in result
    assert (
        "**Most common multilabel combination**: color = yellow, size = short — "
        "2 out of 5 (40.0%)" in result
    )


def test_single_column_does_not_report_a_combination():
    """With only one configured column, no 'most common combination' line
    is added — that concept requires at least two columns."""
    df = pd.DataFrame({"specie": ["a", "b", "a"]})
    config = {"label_columns": ["specie"]}
    result = class_distribution(df, config)
    assert "Most common multilabel combination" not in result
