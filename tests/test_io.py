"""Unit tests for I/O functions."""

import os

import pandas as pd
import pytest

from eda_workbench.io import (
    get_dataset_config,
    load_config,
    load_dataset,
    save_markdown,
)


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


def test_load_config_explicit_path(tmp_path):
    """Loads config from an explicitly given path, ignoring the default candidates."""
    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text("datasets:\n  toy:\n    id_column: pk\n")

    config = load_config(config_path=custom_config)
    assert config == {"datasets": {"toy": {"id_column": "pk"}}}


def test_load_config_explicit_path_missing_returns_empty(tmp_path):
    """Returns an empty dict when the explicitly given path does not exist."""
    missing_path = tmp_path / "does_not_exist.yaml"
    assert load_config(config_path=missing_path) == {}


def test_load_config_falls_back_to_repo_default(tmp_path, monkeypatch):
    """Falls back to the repository's default config when cwd has none."""
    monkeypatch.chdir(tmp_path)  # no config/settings.yaml here

    config = load_config()
    assert "datasets" in config
    assert "wine" in config["datasets"]


def test_load_config_returns_empty_when_no_candidate_exists(tmp_path, monkeypatch):
    """Returns an empty dict when neither the cwd nor the default path has a config file."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "eda_workbench.io._DEFAULT_CONFIG_PATH", tmp_path / "missing.yaml"
    )

    assert load_config() == {}


def test_save_markdown(tmp_path):
    """Saves report content to a markdown file with correct name."""
    path = save_markdown("hello", "data/wine.csv", output_dir=str(tmp_path))
    assert os.path.exists(path)
    assert "wine_report.md" in path
    with open(path) as f:
        assert f.read() == "hello"
