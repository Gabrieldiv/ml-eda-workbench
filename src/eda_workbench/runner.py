"""Runner — processes all CSV files in a folder and generates EDA reports."""

import os

from eda_workbench.analysis import basic_info, missing_values, zero_values
from eda_workbench.io import get_dataset_config, load_config, load_dataset, save_txt

# -----------------------------------------------------------------------------
# ORCHESTRATION
# -----------------------------------------------------------------------------


def _collect_csv_files(data_dir: str) -> list[str]:
    """Return all CSV file paths in a folder, with user-friendly messages if none are found."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Folder '{data_dir}' created. Add your CSV files there and run again.")
        return []

    csv_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")
    ]

    if not csv_files:
        print(
            f"No CSV files found in '{data_dir}'. Add your datasets there and run again."
        )

    return csv_files


def process_folder(data_dir: str = "data", output_dir: str = "output") -> None:
    """Process all CSV files in a folder and generate EDA reports.

    For each CSV found, loads the dataset, runs the analyses,
    and saves the report as a TXT file in the output folder.

    Args:
        data_dir: Path to the folder containing CSV files. Default: "data".
        output_dir: Path to the folder where reports will be saved. Default: "output".
    """
    csv_files = _collect_csv_files(data_dir)
    if not csv_files:
        return

    config = load_config()

    for csv_path in csv_files:
        print(f"\nProcessing: {csv_path}")
        print("-" * 40)

        base_name, _ = os.path.splitext(os.path.basename(csv_path))
        dataset_config = get_dataset_config(base_name, config)

        df = load_dataset(csv_path, dataset_config)
        report = basic_info(df, dataset_config["id_column"])
        report += (
            f"\nInstances with missing values: {missing_values(df)} out of {len(df)}"
        )
        report += f"\n{zero_values(df, dataset_config)}"

        print(report)

        path = save_txt(report, csv_path, output_dir=output_dir)
        print(f"\nReport saved to: {path}")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    process_folder()
