"""I/O utilities — loading datasets and saving reports."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.figure import Figure

from eda_workbench.bias.prepare import as_name_list

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"

_DATASET_DEFAULTS: dict = {
    "separator": ",",
    "encoding": "utf-8",
    "id_column": None,
    "target_column": "target",
    "protected_attributes": [],
    "eval_path": None,
    "predictions_path": None,
    "model_path": None,
    "prediction_column": "y_pred",
    "score_column": "y_score",
    "model_features": None,
}


def _dataset_defaults() -> dict:
    defaults = dict(_DATASET_DEFAULTS)
    defaults["protected_attributes"] = list(_DATASET_DEFAULTS["protected_attributes"])
    return defaults


def load_config(config_path: str | os.PathLike[str] | None = None) -> dict:
    """Load the full YAML configuration file.

    Args:
        config_path: Optional path to the configuration file. If None, tries
            ``config/settings.yaml`` in the current working directory, then
            falls back to the repository root when running from source.

    Returns:
        Dictionary with all configuration settings. Returns an empty dict if no
        config file is found.
    """

    candidates: list[Path] = []
    if config_path is not None:
        candidates.append(Path(config_path))
    else:
        candidates.append(Path("config/settings.yaml"))
        candidates.append(_DEFAULT_CONFIG_PATH)

    for path in candidates:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

    return {}


def get_dataset_config(dataset_name: str, config: dict) -> dict:
    """Return dataset-specific config, or default CSV settings if not registered.

    Args:
        dataset_name: Name of the dataset (e.g. "wine", "titanic_train").
        config: Full configuration dictionary loaded from YAML.

    Returns:
        Dictionary with separator, encoding, id/target columns, protected
        attributes, and optional eval-bundle paths.
    """
    merged = _dataset_defaults()
    datasets = config.get("datasets", {})
    if dataset_name in datasets:
        merged.update(datasets[dataset_name])
    merged["protected_attributes"] = as_name_list(merged.get("protected_attributes"))
    features = merged.get("model_features")
    if features is not None:
        merged["model_features"] = as_name_list(features)
    return merged


def load_dataset(csv_path: str, config: dict) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame.

    Args:
        csv_path: Path to the CSV file to load.
        config: Dictionary with dataset settings (separator, encoding).

    Returns:
        DataFrame with the file data.

    Raises:
        ValueError: If the file does not have a .csv extension.
    """
    if not csv_path.endswith(".csv"):
        raise ValueError(f"File must be .csv, got: {csv_path}")

    return pd.read_csv(csv_path, sep=config["separator"], encoding=config["encoding"])


def save_markdown(content: str, csv_path: str, output_dir: str = "output") -> str:
    """Save report content to a Markdown file.

    The file name is based on the input CSV name.
    Example: "data/wine.csv" generates "output/wine_report.md".

    Args:
        content: Report Markdown content to save.
        csv_path: Path to the source CSV, used to name the report.
        output_dir: Directory where the file will be saved. Default: "output".

    Returns:
        Full path of the generated Markdown file.
    """
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(csv_path)
    base_name, _ = os.path.splitext(file_name)

    path = os.path.join(output_dir, f"{base_name}_report.md")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return path


def save_plot(figure: Figure, csv_path: str, output_dir: str = "output") -> str:
    """Save a matplotlib figure to a PNG file, then close it to free memory.

    The file name is based on the input CSV name.
    Example: "data/wine.csv" generates "output/wine_characteristics.png".

    Args:
        figure: Matplotlib Figure to save.
        csv_path: Path to the source CSV, used to name the file.
        output_dir: Directory where the file will be saved. Default: "output".

    Returns:
        Full path of the generated PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(csv_path)
    base_name, _ = os.path.splitext(file_name)

    path = os.path.join(output_dir, f"{base_name}_characteristics.png")

    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return path


def save_correlation_outputs(
    figures: dict[str, Figure],
    correlation_matrix: pd.DataFrame,
    base_name: str,
    output_dir: str = "output",
) -> str:
    """Save correlation plots and the correlation matrix into a per-dataset folder.

    Creates ``{output_dir}/{base_name}_correlation/`` and writes
    ``correlation_matrix.csv`` (when the matrix has data), plus
    ``correlation_heatmap.png`` and/or ``correlation_violin.png`` for
    whichever figures are present. Each figure is closed after saving.

    Args:
        figures: Dict with optional "heatmap" and/or "violin" Figure objects.
        correlation_matrix: The full correlation matrix to export as CSV.
        base_name: Dataset name (without extension), used to name the folder.
        output_dir: Directory where the folder is created. Default: "output".

    Returns:
        Name of the created subfolder (e.g. "wine_correlation"), for building
        a relative Markdown image path. Empty string when there is nothing
        to save (no figures and an empty matrix).
    """
    if not figures and correlation_matrix.empty:
        return ""

    folder_name = f"{base_name}_correlation"
    folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    if not correlation_matrix.empty:
        correlation_matrix.to_csv(os.path.join(folder_path, "correlation_matrix.csv"))

    if "heatmap" in figures:
        figures["heatmap"].savefig(
            os.path.join(folder_path, "correlation_heatmap.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figures["heatmap"])

    if "violin" in figures:
        figures["violin"].savefig(
            os.path.join(folder_path, "correlation_violin.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figures["violin"])

    return folder_name
