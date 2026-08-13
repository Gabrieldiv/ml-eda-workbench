"""Unit tests for analysis_provenance."""

import re

from eda_workbench.analysis import analysis_provenance


def test_includes_dataset_name():
    """Reports the dataset name."""
    result = analysis_provenance("wine.csv", 0.9, 30, 1.5, 10)
    assert "**Dataset**: wine.csv" in result


def test_includes_all_thresholds_used():
    """Reports every threshold value passed in, for reproducibility."""
    result = analysis_provenance("data.csv", 0.85, 25, 2.0, 5)
    assert "**Correlation redundancy threshold**: 0.85" in result
    assert "**Minimum pairwise N for correlation**: 25" in result
    assert "**Outlier whisker multiplier**: 2.0" in result
    assert "**Low-cardinality threshold**: 5" in result


def test_includes_package_versions():
    """Reports the versions of pandas, numpy, and matplotlib in use."""
    import matplotlib
    import numpy as np
    import pandas as pd

    result = analysis_provenance("data.csv", 0.9, 30, 1.5, 10)
    assert f"pandas {pd.__version__}" in result
    assert f"numpy {np.__version__}" in result
    assert f"matplotlib {matplotlib.__version__}" in result


def test_includes_generation_timestamp_in_expected_format():
    """Reports a generation timestamp with date, time, offset, and timezone name."""
    result = analysis_provenance("data.csv", 0.9, 30, 1.5, 10)
    assert re.search(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{4} \(America/Sao_Paulo\)",
        result,
    )
