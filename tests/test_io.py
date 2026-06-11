"""Unit tests for I/O functions."""

import os
import pandas as pd
import pytest
from eda_workbench.io import load_config, load_dataset, save_txt

def test_load_dataset_valid_csv(tmp_path):
    """Loads a valid CSV and returns a DataFrame."""
    csv = tmp_path / "test.csv"
    csv.write_text("a,b\n1,2\n3,4")
    config = {"separator": ",", "encoding": "utf-8"}
    df = load_dataset(str(csv), config)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)

def test_load_config_unknown_dataset():
    """Raises KeyError for datasets not configured in the YAML."""
    with pytest.raises(KeyError):
        load_config("dataset_que_nao_existe")

def test_load_dataset_invalid_extension():
    """Raises ValueError for non-CSV files."""
    with pytest.raises(ValueError):
        load_dataset("file.txt", {})

def test_load_config_valid_dataset():
    """Returns config dict for a known dataset."""
    config = load_config("wine")
    assert config["separator"] == ","
    assert config["encoding"] == "utf-8"

def test_save_txt(tmp_path):
    """Saves report content to a TXT file with correct name."""
    path = save_txt("hello", "data/wine.csv", output_dir=str(tmp_path))
    assert os.path.exists(path)
    assert "wine_report.txt" in path
    with open(path) as f:
        assert f.read() == "hello"