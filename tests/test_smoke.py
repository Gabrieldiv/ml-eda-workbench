"""Smoke tests — verify all core libraries are importable and functional.

These tests act as an environment health check. If they pass, the student
knows the virtual environment is set up correctly and every dependency is
available. They are intentionally minimal: no business logic is tested here.
"""

import matplotlib
import numpy as np
import pandas as pd
import plotly
import scipy
import seaborn
import sklearn  # noqa: F401

# ---------------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------------


def test_numpy_importable():
    assert hasattr(np, "__version__")


def test_numpy_basic_operations():
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.sum() == 15
    assert arr.mean() == 3.0
    assert arr.std() > 0


def test_numpy_random_generator():
    rng = np.random.default_rng(seed=0)
    sample = rng.normal(size=100)
    assert sample.shape == (100,)


# ---------------------------------------------------------------------------
# Pandas
# ---------------------------------------------------------------------------


def test_pandas_importable():
    assert hasattr(pd, "__version__")


def test_pandas_dataframe_creation():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})
    assert df.shape == (3, 3)
    assert list(df.columns) == ["a", "b", "c"]


def test_pandas_missing_value_detection():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    assert df["x"].isna().sum() == 1


# ---------------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------------


def test_matplotlib_importable():
    assert hasattr(matplotlib, "__version__")


def test_matplotlib_figure_creation():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# Seaborn
# ---------------------------------------------------------------------------


def test_seaborn_importable():
    assert hasattr(seaborn, "__version__")


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------


def test_plotly_importable():
    assert hasattr(plotly, "__version__")


# ---------------------------------------------------------------------------
# Scipy
# ---------------------------------------------------------------------------


def test_scipy_importable():
    assert hasattr(scipy, "__version__")


def test_scipy_stats():
    from scipy import stats

    result = stats.ttest_1samp([1.0, 2.0, 3.0, 4.0, 5.0], popmean=3.0)
    assert hasattr(result, "pvalue")


# ---------------------------------------------------------------------------
# Scikit-learn
# ---------------------------------------------------------------------------


def test_sklearn_importable():
    assert hasattr(sklearn, "__version__")


def test_sklearn_train_test_split():
    from sklearn.model_selection import train_test_split

    X = np.arange(100).reshape(50, 2)  # noqa: N806
    y = np.zeros(50)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )  # noqa: N806
    assert len(X_train) == 40
    assert len(X_test) == 10


# ---------------------------------------------------------------------------
# eda_workbench package
# ---------------------------------------------------------------------------


def test_package_importable():
    import eda_workbench

    assert hasattr(eda_workbench, "Settings")


def test_settings_defaults():
    from eda_workbench.config import Settings

    cfg = Settings()
    assert cfg.experiment.random_seed == 42
    assert cfg.experiment.test_size == 0.2
    assert cfg.data.target_column == "target"


def test_settings_from_yaml(settings):
    assert settings.experiment.random_seed == 42
    assert settings.data.path == "data/"


def test_sample_fixture_shape(sample_df):
    assert sample_df.shape == (50, 5)
    assert "target" in sample_df.columns


def test_sample_fixture_has_missing_values(sample_df):
    assert sample_df["feature_c"].isna().sum() > 0
