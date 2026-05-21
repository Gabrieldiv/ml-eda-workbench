"""Shared pytest fixtures for the EDA workbench test suite.

All fixtures defined here are automatically available in every test file —
no imports needed. Just declare the fixture name as a function parameter.

Example::

    def test_my_function(sample_df, settings):
        assert sample_df.shape[0] == 50
        assert settings.experiment.random_seed == 42
"""

import numpy as np
import pandas as pd
import pytest

from eda_workbench.config import Settings


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small synthetic DataFrame for unit tests.

    Contains:
    - ``feature_a``: 50 normally-distributed float values
    - ``feature_b``: 50 uniformly-distributed float values
    - ``feature_c``: 50 integer values with some missing entries
    - ``category``:  50 categorical string labels
    - ``target``:    50 binary integer labels (0 or 1)
    """
    rng = np.random.default_rng(seed=42)
    n = 50

    feature_c = rng.integers(0, 100, n).astype(float)
    feature_c[rng.choice(n, size=5, replace=False)] = np.nan

    return pd.DataFrame(
        {
            "feature_a": rng.normal(loc=0.0, scale=1.0, size=n),
            "feature_b": rng.uniform(low=0.0, high=10.0, size=n),
            "feature_c": feature_c,
            "category": rng.choice(["cat", "dog", "bird"], size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )


@pytest.fixture
def settings() -> Settings:
    """A :class:`~eda_workbench.config.Settings` object loaded from
    ``config/settings.yaml``."""
    return Settings.from_yaml()
