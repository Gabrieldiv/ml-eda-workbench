"""Unit tests for representational coverage on a profiled table."""

import pandas as pd

from eda_workbench.bias.coverage import compute_coverage
from eda_workbench.bias.prepare import resolve_thresholds

_THRESHOLDS = resolve_thresholds(
    {
        "min_group_count": 20,
        "min_group_share": 0.15,
        "sparse_intersection_share": 0.08,
    }
)


def test_coverage_skips_without_protected_attributes():
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    result = compute_coverage(frame, {"protected_attributes": []}, _THRESHOLDS)
    assert result.skipped is not None
    assert "no protected attributes" in result.skipped.lower()
    assert result.attributes == ()


def test_coverage_counts_imbalance_and_sparse_intersections():
    frame = pd.DataFrame(
        {
            "Sex": ["male"] * 90 + ["female"] * 10,
            "Pclass": [3] * 80 + [1] * 15 + [2] * 5,
        }
    )
    result = compute_coverage(
        frame,
        {"protected_attributes": ["Sex", "Pclass"]},
        _THRESHOLDS,
    )
    assert result.skipped is None
    assert result.n_rows == 100

    by_name = {item.attribute: item for item in result.attributes}
    sex = {group.group: group for group in by_name["Sex"].groups}
    assert sex["male"].count == 90
    assert sex["female"].count == 10
    assert sex["female"].underrepresented is True
    assert by_name["Sex"].imbalance_ratio == 9.0

    pclass = {group.group: group for group in by_name["Pclass"].groups}
    assert pclass["2"].underrepresented is True
    assert any(item.sparse for item in result.intersections)


def test_coverage_notes_missing_attribute_columns():
    frame = pd.DataFrame({"Sex": ["male", "female"]})
    result = compute_coverage(
        frame,
        {"protected_attributes": ["Sex", "Pclass"]},
        _THRESHOLDS,
    )
    assert result.skipped is None
    assert any("Pclass" in note for note in result.notes)
    assert result.attributes[0].attribute == "Sex"
