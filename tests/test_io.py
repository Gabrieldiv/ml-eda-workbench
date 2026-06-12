"""Unit tests for I/O functions."""

import os

import pandas as pd
import pytest

from eda_workbench.io import get_dataset_config, load_config, load_dataset, save_txt


def test_load_dataset_valid_csv(tmp_path):
    """Loads a valid CSV and returns a DataFrame."""
    csv = tmp_path / "test.csv"
    csv.write_text("a,b\n1,2\n3,4")
    config = {"separator": ",", "encoding": "utf-8"}
    df = load_dataset(str(csv), config)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_load_dataset_invalid_extension():
    """Raises ValueError for non-CSV files."""
    with pytest.raises(ValueError):
        load_dataset("file.txt", {})


def test_load_config_valid_dataset():
    """Returns dataset-specific config for registered datasets."""
    config = load_config()
    dataset_config = get_dataset_config("wine", config)
    assert dataset_config["separator"] == ","
    assert dataset_config["encoding"] == "utf-8"


def test_load_config_unknown_dataset():
    """Returns default config for datasets not registered in the YAML."""
    config = load_config()
    dataset_config = get_dataset_config("non_existent_dataset", config)
    assert dataset_config["separator"] == ","
    assert dataset_config["id_column"] is None


def test_save_txt(tmp_path):
    """Saves report content to a TXT file with correct name."""
    path = save_txt("hello", "data/wine.csv", output_dir=str(tmp_path))
    assert os.path.exists(path)
    assert "wine_report.txt" in path
    with open(path) as f:
        assert f.read() == "hello"
