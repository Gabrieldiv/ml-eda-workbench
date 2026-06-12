"""I/O utilities — loading datasets and saving reports."""

import os
import pandas as pd
import yaml


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load the full YAML configuration file.

    Args:
        config_path: Path to the configuration file. Default: "config/settings.yaml".

    Returns:
        Dictionary with all configuration settings.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dataset_config(dataset_name: str, config: dict) -> dict:
    """Return dataset-specific config, or default CSV settings if not registered.

    Args:
        dataset_name: Name of the dataset (e.g. "wine", "titanic_train").
        config: Full configuration dictionary loaded from YAML.

    Returns:
        Dictionary with separator, encoding, id_column and target_column.
    """
    datasets = config.get("datasets", {})
    if dataset_name in datasets:
        return dict(datasets[dataset_name])

    return {
        "separator": ",",
        "encoding": "utf-8",
        "id_column": None,
        "target_column": "target",
    }


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


def save_txt(content: str, csv_path: str, output_dir: str = "output") -> str:
    """Save report content to a TXT file.

    The file name is based on the input CSV name.
    Example: "data/wine.csv" generates "output/wine_report.txt".

    Args:
        content: Report text to save.
        csv_path: Path to the source CSV, used to name the report.
        output_dir: Directory where the file will be saved. Default: "output".

    Returns:
        Full path of the generated TXT file.
    """
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(csv_path)
    base_name, _ = os.path.splitext(file_name)

    path = os.path.join(output_dir, f"{base_name}_report.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return path