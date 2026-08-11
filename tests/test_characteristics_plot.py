"""Unit tests for characteristics_plot."""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from eda_workbench.analysis import characteristics_plot


def test_empty_dataset():
    """Returns a message and no figure for an empty dataset."""
    df = pd.DataFrame()
    config = {"id_column": None, "target_column": None}
    text, figure = characteristics_plot(df, config)
    assert "**Empty dataset**" in text
    assert figure is None


def test_only_id_and_target_columns():
    """Returns a message and no figure when only ID and target remain."""
    df = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
    config = {"id_column": "id", "target_column": "target"}
    text, figure = characteristics_plot(df, config)
    assert "no analyzable data remains" in text
    assert figure is None


def test_no_numeric_characteristics():
    """Returns a message and no figure when there are no numeric columns."""
    df = pd.DataFrame({"nome": ["a", "b"]})
    config = {"id_column": None, "target_column": None}
    text, figure = characteristics_plot(df, config)
    assert "**No numeric characteristics found**" in text
    assert figure is None


def test_column_with_no_valid_data_is_excluded():
    """Excludes a fully-NaN column from the plot but still plots the rest."""
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1, 2]})
    config = {"id_column": None, "target_column": None}
    text, figure = characteristics_plot(df, config)
    assert "**Characteristics with no valid data**: a" in text
    assert isinstance(figure, Figure)


def test_all_columns_with_no_valid_data():
    """Returns a message and no figure when every numeric column is empty."""
    df = pd.DataFrame({"a": [np.nan], "b": [np.nan]})
    config = {"id_column": None, "target_column": None}
    text, figure = characteristics_plot(df, config)
    assert "**No numeric characteristics with valid data**" in text
    assert figure is None


def test_skips_non_numeric_columns_but_still_plots():
    """Reports skipped non-numeric columns while still generating a figure."""
    df = pd.DataFrame({"nome": ["a", "b"], "idade": [10, 20]})
    config = {"id_column": None, "target_column": None}
    text, figure = characteristics_plot(df, config)
    assert "**Skipped non-numeric characteristics**: nome" in text
    assert isinstance(figure, Figure)


def test_generates_valid_figure_for_normal_case():
    """Generates a matplotlib Figure for a normal, fully numeric dataset."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    config = {"id_column": None, "target_column": None}
    text, figure = characteristics_plot(df, config)
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
