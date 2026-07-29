import pandas as pd

from eda_workbench.analysis import missing_values


def test_missing_values_none():
    """Returns 0 when no missing values exist."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    assert missing_values(df) == 0


def test_missing_values_some():
    """Counts only rows with at least one missing value."""
    df = pd.DataFrame({"a": [1, None, 3], "b": [4.0, 5.0, None]})
    assert missing_values(df) == 2