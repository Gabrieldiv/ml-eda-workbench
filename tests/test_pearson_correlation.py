"""Unit tests for pearson_correlation."""

import pandas as pd

from eda_workbench.analysis import pearson_correlation


def test_skips_non_numeric_columns():
    """Reports non-numeric columns as skipped, without breaking the analysis."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "nome": ["Ana", "Bruno", "Carlos"],
            "idade": [25, 30, 35],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result = pearson_correlation(df, config, limit=0.9)
    assert "**Skipped non-numeric characteristics**" in result
    assert "nome" in result


def test_no_numeric_columns():
    """Reports that no numeric characteristics were found."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "nome": ["Ana", "Bruno", "Carlos"],
            "cargo": ["analista", "gerente", "servente"],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result = pearson_correlation(df, config, limit=0.9)
    assert "**No numeric characteristics found**" in result


def test_finds_redundant_pair():
    """Flags a pair with correlation above the limit and suggests a column to drop."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a": [1, 2, 3, 4],
            "b": [2, 4, 6, 8],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result = pearson_correlation(df, config, limit=0.9)
    assert "a <-> b" in result
    assert "1.000" in result


def test_no_redundant_pairs():
    """Reports that all correlations are below the limit."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a": [1, 5, 2, 8],
            "b": [4, 2, 9, 1],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result = pearson_correlation(df, config, limit=0.9)
    assert "**All correlations below the defined limit (0.9)**" in result


def test_handles_constant_column():
    """Flags columns where correlation cannot be computed, without crashing."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a": [1, 2, 3, 4],
            "constante": [5, 5, 5, 5],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result = pearson_correlation(df, config, limit=0.9)
    assert "Could not compute correlation (not enough variation/data):" in result
    assert "`a` <-> `constante`" in result


def test_suggests_dropping_higher_mean_correlation_column():
    """Between two redundant columns, suggests dropping the one more correlated overall."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "a": [1, 5, 2, 8, 3],
            "b": [1.05, 5.1, 2.1, 8.2, 3.1],
            "c": [1.1, 5.3, 2.2, 8.5, 3.3],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result = pearson_correlation(df, config, limit=0.9)
    assert "| a <-> b | 1.000       | b" in result


def test_blank_line_separates_skipped_columns_from_table():
    """Adds a blank line between the skipped-columns note and the table
    when both are present, so the table renders correctly."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "nome": ["Ana", "Bruno", "Carlos", "Diana"],
            "a": [1, 2, 3, 4],
            "b": [2, 4, 6, 8],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result = pearson_correlation(df, config, limit=0.9)

    assert "**Skipped non-numeric characteristics**" in result
    assert "| a <-> b" in result
    assert "**Skipped non-numeric characteristics**: nome\n\n| Pair" in result


def test_blank_line_separates_skipped_columns_from_nan_pairs():
    """Adds a blank line between the skipped-columns note and the
    'could not compute' section when both are present."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "nome": ["Ana", "Bruno", "Carlos", "Diana"],
            "a": [1, 2, 3, 4],
            "constante": [5, 5, 5, 5],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result = pearson_correlation(df, config, limit=0.9)

    assert "**Skipped non-numeric characteristics**" in result
    assert "Could not compute correlation" in result
    assert (
        "**Skipped non-numeric characteristics**: nome\n\n"
        "**Could not compute correlation" in result
    )
