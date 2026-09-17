"""Unit tests for I/O functions."""

import os

import pandas as pd
import pytest

from eda_workbench.io import (
    get_dataset_config,
    load_config,
    load_dataset,
    save_correlation_outputs,
    save_markdown,
    save_plot,
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
    assert dataset_config["protected_attributes"] == []


def test_load_config_unknown_dataset():
    """Returns default config for datasets not registered in the YAML."""
    config = load_config()
    dataset_config = get_dataset_config("non_existent_dataset", config)
    assert dataset_config["separator"] == ","
    assert dataset_config["id_column"] is None
    assert dataset_config["protected_attributes"] == []
    assert dataset_config["eval_path"] is None
    assert dataset_config["predictions_path"] is None
    assert dataset_config["model_path"] is None
    assert dataset_config["prediction_column"] == "y_pred"


def test_get_dataset_config_treats_scalar_names_as_one_item():
    config = {
        "datasets": {
            "toy": {
                "protected_attributes": "Sex",
                "model_features": "Age",
            }
        }
    }
    dataset_config = get_dataset_config("toy", config)
    assert dataset_config["protected_attributes"] == ["Sex"]
    assert dataset_config["model_features"] == ["Age"]


def test_load_config_titanic_protected_attributes():
    """Titanic example lists Sex and Pclass; eval paths stay optional."""
    config = load_config()
    dataset_config = get_dataset_config("titanic_train", config)
    assert dataset_config["protected_attributes"] == ["Sex", "Pclass"]
    assert dataset_config["eval_path"] is None
    assert dataset_config["predictions_path"] is None
    assert dataset_config["model_path"] is None


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


def test_save_plot_closes_figure_after_saving(tmp_path):
    """Closes the figure after saving to avoid accumulating memory."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    open_before = len(plt.get_fignums())
    save_plot(fig, "data/wine.csv", output_dir=str(tmp_path))
    open_after = len(plt.get_fignums())

    assert open_after == open_before - 1


def test_save_correlation_outputs_skips_csv_when_matrix_empty_but_figure_present(
    tmp_path,
):
    """Does not write correlation_matrix.csv when the matrix is empty,
    even though a figure was still provided."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])

    folder = save_correlation_outputs(
        {"heatmap": fig}, pd.DataFrame(), "wine", output_dir=str(tmp_path)
    )
    saved_files = os.listdir(tmp_path / folder)

    assert "correlation_matrix.csv" not in saved_files
    assert "correlation_heatmap.png" in saved_files


def test_save_correlation_outputs_saves_violin_figure(tmp_path):
    """Saves correlation_violin.png when only the violin figure is present,
    without a heatmap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.violinplot([[1, 2, 3]])
    matrix = pd.DataFrame({"a": [1.0, 0.5], "b": [0.5, 1.0]}, index=["a", "b"])

    folder = save_correlation_outputs(
        {"violin": fig}, matrix, "wine", output_dir=str(tmp_path)
    )
    saved_files = os.listdir(tmp_path / folder)

    assert "correlation_violin.png" in saved_files
    assert "correlation_heatmap.png" not in saved_files
    assert "correlation_matrix.csv" in saved_files
