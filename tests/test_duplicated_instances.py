import pandas as pd

from eda_workbench.analysis import duplicated_instances

def test_duplicated_instances_none():
    """Returns 'None' when every instance is unique."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    config = {"id_column": None, "target_column": None}
    result = duplicated_instances(df, config)
    assert "None" in result


def test_duplicated_instances_one_group():
    """Finds a single pair of identical instances, ignoring ID and target."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "a": [10, 10, 99],
            "b": [20, 20, 88],
            "target": [0, 1, 1],
        }
    )
    config = {"id_column": "id", "target_column": "target"}
    result = duplicated_instances(df, config)
    assert "2 instances - IDs/row index: 1, 2" in result


def test_duplicated_instances_uses_row_index_without_id():
    """Falls back to row position when there is no ID column."""
    df = pd.DataFrame({"a": [10, 10], "b": [20, 20]})
    config = {"id_column": None, "target_column": None}
    result = duplicated_instances(df, config)
    assert "2 instances - IDs/row index: 0, 1" in result


def test_duplicated_instances_multiple_groups():
    """Detects more than one group of duplicates at the same time."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "a": [10, 10, 99, 5, 5],
            "b": [20, 20, 88, 6, 6],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result = duplicated_instances(df, config)
    assert "2 instances - IDs/row index: 1, 2" in result
    assert "2 instances - IDs/row index: 4, 5" in result


def test_duplicated_instances_same_features_different_targets():
    """Still counts as duplicate even if the target differs (target is ignored)."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "a": [10, 10],
            "target": [0, 1],
        }
    )
    config = {"id_column": "id", "target_column": "target"}
    result = duplicated_instances(df, config)
    assert "2 instances - IDs/row index: 1, 2" in result
    assert "2 instances - different targets: 0, 1" in result


def test_duplicated_instances_empty_dataframe():
    """Returns 'None' when the DataFrame has no rows."""
    df = pd.DataFrame({"a": [], "b": []})
    config = {"id_column": None, "target_column": None}
    result = duplicated_instances(df, config)
    assert "None" in result


def test_duplicated_instances_nan_values_match():
    """Treats matching NaN values as equal when comparing rows."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "a": [float("nan"), float("nan")],
            "b": [5, 5],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result = duplicated_instances(df, config)
    assert "2 instances - IDs/row index: 1, 2" in result


def test_duplicated_instances_same_target_no_conflict_flag():
    """Does not flag a conflict when duplicated instances share the same target."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "a": [10, 10],
            "target": [0, 0],
        }
    )
    config = {"id_column": "id", "target_column": "target"}
    result = duplicated_instances(df, config)
    assert "different targets" not in result
