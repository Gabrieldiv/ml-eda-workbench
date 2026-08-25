"""Unit tests for fairlearn metrics on a separate predictions table."""

import pandas as pd
import pytest

from eda_workbench.bias.metric_audit import audit_metrics
from eda_workbench.bias.prepare import (
    as_name_list,
    protected_attributes,
    resolve_thresholds,
)

_THRESHOLDS = resolve_thresholds(
    {"rate_difference_threshold": 0.1, "disparate_impact_threshold": 0.8}
)
_CONFIG = {
    "protected_attributes": ["sex"],
    "target_column": "target",
    "prediction_column": "y_pred",
    "score_column": "y_score",
}


def _predictions_with_gap() -> pd.DataFrame:
    n = 50
    return pd.DataFrame(
        {
            "y_true": [1, 0] * (n // 2) + [1, 0] * (n // 2),
            "y_pred": [1] * n + [0] * n,
            "y_score": [0.9] * n + [0.1] * n,
            "sex": ["f"] * n + ["m"] * n,
        }
    )


def test_metrics_use_predictions_table_not_profiled_csv():
    """Metrics run on a standalone predictions frame; no profiled CSV is used."""
    result = audit_metrics(_predictions_with_gap(), _CONFIG, _THRESHOLDS)
    assert result.skipped is None
    assert result.binary_target is True
    assert result.n_rows == 100
    summary = result.summaries[0]
    assert summary.demographic_parity_difference is not None
    assert abs(summary.demographic_parity_difference) > 0.5
    assert summary.flags
    assert all(
        row.actual_positive_rate == pytest.approx(0.5) for row in result.by_group
    )


def test_metrics_skip_without_protected_attributes():
    result = audit_metrics(
        _predictions_with_gap(),
        {**_CONFIG, "protected_attributes": []},
        _THRESHOLDS,
    )
    assert result.skipped is not None
    assert "protected attributes" in result.skipped.lower()


def test_metrics_skip_without_prediction_column():
    frame = pd.DataFrame({"y_true": [0, 1], "sex": ["f", "m"]})
    result = audit_metrics(frame, _CONFIG, _THRESHOLDS)
    assert result.skipped is not None
    assert "y_pred" in result.skipped


def test_metrics_non_binary_reports_accuracy_only():
    frame = pd.DataFrame(
        {
            "y_true": [0, 1, 2, 0, 1, 2],
            "y_pred": [0, 1, 1, 0, 2, 2],
            "sex": ["f", "f", "f", "m", "m", "m"],
        }
    )
    result = audit_metrics(frame, _CONFIG, _THRESHOLDS)
    assert result.skipped is None
    assert result.binary_target is False
    assert result.summaries[0].demographic_parity_difference is None
    assert all(row.accuracy is not None for row in result.by_group)


def test_metrics_string_labels_are_not_treated_as_binary():
    frame = pd.DataFrame(
        {
            "y_true": ["yes", "no"] * 3,
            "y_pred": ["yes", "yes", "no", "no", "yes", "no"],
            "sex": ["f", "f", "f", "m", "m", "m"],
        }
    )
    result = audit_metrics(frame, _CONFIG, _THRESHOLDS)
    assert result.skipped is None
    assert result.binary_target is False
    assert result.summaries[0].demographic_parity_difference is None


def test_protected_attributes_treats_scalar_string_as_one_name():
    assert as_name_list("Sex") == ["Sex"]
    assert protected_attributes({"protected_attributes": "Sex"}) == ["Sex"]
    assert as_name_list(["Sex", "Pclass"]) == ["Sex", "Pclass"]
