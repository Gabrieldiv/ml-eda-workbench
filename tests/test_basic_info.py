"""Unit tests for basic_info."""

import pandas as pd

from eda_workbench.analysis import basic_info


def test_basic_info_with_id_and_target():
    """Reports ID and target columns, excluding both from the characteristic count."""
    df = pd.DataFrame({"id": [1, 2, 3], "a": [1, 2, 3], "target": [0, 1, 0]})
    config = {"id_column": "id", "target_column": "target"}
    result = basic_info(df, config)
    assert "**ID column**: `id`" in result
    assert "**Target column**: `target`" in result
    assert "**Rows**: 3" in result
    assert "Characteristics (excluding ID and target)**: 1" in result


def test_basic_info_only_id_column():
    """Excludes only the ID column when there is no target configured."""
    df = pd.DataFrame({"id": [1, 2, 3], "a": [1, 2, 3], "b": [4, 5, 6]})
    config = {"id_column": "id", "target_column": None}
    result = basic_info(df, config)
    assert "**ID column**: `id`" in result
    assert "**No Target column detected**" in result
    assert "Characteristics (excluding ID)**: 2" in result


def test_basic_info_only_target_column():
    """Excludes only the target column when there is no ID configured."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "specie": ["x", "y", "z"]})
    config = {"id_column": None, "target_column": "specie"}
    result = basic_info(df, config)
    assert "**No ID column detected**" in result
    assert "**Target column**: `specie`" in result
    assert "Characteristics (excluding target)**: 2" in result


def test_basic_info_neither_id_nor_target():
    """States that ID and target are not configured, counting all columns."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    config = {"id_column": None, "target_column": None}
    result = basic_info(df, config)
    assert "**No ID column detected**" in result
    assert "**No Target column detected**" in result
    assert "no ID or target column found in dataset" in result
    assert "**: 2" in result


def test_basic_info_target_configured_but_missing():
    """States that the target is configured but absent, without excluding it."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    config = {"id_column": None, "target_column": "target"}
    result = basic_info(df, config)
    assert "**Target column `target` configured but not found in dataset**" in result
    assert "**: 2" in result  # nada foi excluído, continuam as 2 colunas originais


def test_basic_info_id_configured_but_missing():
    """States that the ID is configured but absent, without excluding it."""
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    config = {"id_column": "id_que_nao_existe", "target_column": "target"}
    result = basic_info(df, config)
    assert (
        "**ID column `id_que_nao_existe` configured but not found in dataset**"
        in result
    )
    assert "**Target column**: `target`" in result
    assert (
        "**: 1" in result
    )  # só 'a' sobra: target foi excluído de verdade, id não existia
