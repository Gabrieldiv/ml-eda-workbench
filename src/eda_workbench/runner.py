"""Runner — processes all CSV files in a folder and generates EDA reports."""

import os
from eda_workbench.io import load_config, load_dataset, save_txt
from eda_workbench.analysis import basic_info, missing_values

# -----------------------------------------------------------------------------
# ORCHESTRATION
# -----------------------------------------------------------------------------

def process_folder(data_dir: str = "data") -> None:
    """Process all CSV files in a folder and generate EDA reports.

    For each CSV found, loads the dataset, runs the analyses,
    and saves the report as a TXT file in the output folder.

    Args:
        data_dir: Path to the folder containing CSV files. Default: "data".
    """
    csv_files = []

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Folder '{data_dir}' created. Put your .csv files there and run again.")
        return

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            csv_files.append(os.path.join(data_dir, file))

    for csv_path in csv_files:
        print(f"\nProcessing: {csv_path}")
        print("-" * 40)

        base_name, _ = os.path.splitext(os.path.basename(csv_path))
        config = load_config(base_name)

        df = load_dataset(csv_path, config)
        report = basic_info(df, config["coluna_id"])
        report += f"\nInstances with missing value: {missing_values(df)} out of {len(df)}"

        print(report)

        path = save_txt(report, csv_path)
        print(f"\nReport saved to: {path}")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    process_folder()