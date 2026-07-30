"""Unit tests for basic_info."""

import pandas as pd

from eda_workbench.analysis import basic_info


def test_basic_info_with_id_column():
    """Reports ID column name when detected."""
    df = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    result = basic_info(df, id_column="id")
    assert "**ID column**: `id`" in result
    assert "**Rows**: 2" in result
    assert "**Columns**: 2" in result


def test_basic_info_without_id_column():
    """Reports no ID column when none is provided."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    result = basic_info(df, id_column=None)
    assert "**No ID column detected**" in result
