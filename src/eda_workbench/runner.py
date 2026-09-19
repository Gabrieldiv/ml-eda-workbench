"""Runner — processes all CSV files in a folder and generates EDA reports."""

import os

from fpdf import FPDF

from eda_workbench.analysis import (
    analysis_provenance,
    basic_info,
    characteristics_plot,
    class_distribution,
    duplicated_instances,
    missing_values,
    outliers_instances,
    pearson_correlation,
    zero_values,
)
from eda_workbench.bias.identify import identify_bias
from eda_workbench.io import (
    get_dataset_config,
    load_config,
    load_dataset,
    save_correlation_outputs,
    save_json,
    save_markdown,
    save_pdf,
    save_plot,
)
from eda_workbench.printout.json_exit import render_missing_values_json
from eda_workbench.printout.markdown_exit import render_missing_values
from eda_workbench.printout.pdf_exit import render_missing_values_pdf
from eda_workbench.reporting.bias import render_bias_identification

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
    and saves the report as a Markdown file in the output folder.
    Additional formats (JSON, PDF) are generated alongside it when
    configured via ``output.extra_formats`` in settings.yaml — currently
    covering only the sections already migrated to the structured
    dataclass pattern (Missing Values so far).

    Args:
        data_dir: Path to the folder containing CSV files. Default: "data".
        output_dir: Path to the folder where reports will be saved. Default: "output".
    """
    csv_files = _collect_csv_files(data_dir)
    if not csv_files:
        return

    config = load_config()
    limit = config.get("correlation", {}).get("drop_threshold", 0.9)
    min_pairwise_n = config.get("correlation", {}).get("min_pairwise_n", 30)
    plot_mode = config.get("correlation", {}).get("plot_mode", "both")
    whisker_multiplier = config.get("outliers", {}).get("outlier_factor", 1.5)
    low_cardinality_threshold = config.get("data_types", {}).get(
        "low_cardinality_threshold", 10
    )
    default_class_dist_threshold = config.get("class_distribution", {}).get(
        "high_cardinality_threshold", 20
    )
    extra_formats = config.get("output", {}).get("extra_formats", [])

    for csv_path in csv_files:
        print(f"\nProcessing: {csv_path}")
        print("-" * 40)

        base_name, _ = os.path.splitext(os.path.basename(csv_path))
        dataset_config = get_dataset_config(base_name, config)

        df = load_dataset(csv_path, dataset_config)

        class_dist_threshold = dataset_config.get(
            "high_cardinality_threshold", default_class_dist_threshold
        )

        report = f"# EDA Report — {base_name}\n\n"
        report += analysis_provenance(
            base_name,
            limit,
            min_pairwise_n,
            whisker_multiplier,
            low_cardinality_threshold,
        )

        report += f"\n\n{basic_info(df, dataset_config)}"

        missing_values_result = missing_values(df)
        report += f"\n\n{render_missing_values(missing_values_result)}"

        if "json" in extra_formats:
            report_json = {
                "dataset": base_name,
                "missing_values": render_missing_values_json(missing_values_result),
            }
            json_path = save_json(report_json, csv_path, output_dir=output_dir)
            print(f"JSON report saved to: {json_path}")

        if "pdf" in extra_formats:
            pdf_doc = FPDF()
            pdf_doc.add_page()
            render_missing_values_pdf(pdf_doc, missing_values_result)
            pdf_path = save_pdf(pdf_doc, csv_path, output_dir=output_dir)
            print(f"PDF report saved to: {pdf_path}")

        report += f"\n\n{zero_values(df, dataset_config)}"
        report += f"\n\n{duplicated_instances(df, dataset_config)}"
        report += f"\n\n{class_distribution(df, dataset_config, class_dist_threshold)}"

        corr_text, corr_figures, corr_matrix = pearson_correlation(
            df,
            dataset_config,
            limit,
            min_pairwise_n,
            low_cardinality_threshold,
            plot_mode,
        )
        report += f"\n\n{corr_text}"

        corr_folder = save_correlation_outputs(
            corr_figures, corr_matrix, base_name, output_dir=output_dir
        )
        if "heatmap" in corr_figures:
            report += (
                f"\n\n![Pearson Correlation Heatmap]"
                f"({corr_folder}/correlation_heatmap.png)"
            )
        if "violin" in corr_figures:
            report += (
                f"\n\n![Pearson Correlation Violin Plot]"
                f"({corr_folder}/correlation_violin.png)"
            )

        bias_cfg = config.get("bias", {})
        bias_result = identify_bias(
            df,
            dataset_config,
            bias_cfg,
            profiled_path=csv_path,
            random_seed=config.get("experiment", {}).get("random_seed", 42),
        )
        report += "\n\n" + render_bias_identification(
            bias_result,
            detail=bias_cfg.get("report_detail", "summary"),
            target_column=dataset_config.get("target_column"),
            grouping_columns=dataset_config.get("protected_attributes") or [],
            thresholds=bias_cfg,
        )

        report += f"\n\n{outliers_instances(df, dataset_config, whisker_multiplier, low_cardinality_threshold)}"

        plot_text, figure = characteristics_plot(df, dataset_config)
        report += f"\n\n{plot_text}"
        if figure is not None:
            image_path = save_plot(figure, csv_path, output_dir=output_dir)
            image_name = os.path.basename(image_path)
            report += f"\n\n![Characteristics Value Ranges]({image_name})"

        print(report)

        path = save_markdown(report, csv_path, output_dir=output_dir)
        print(f"\nReport saved to: {path}")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    process_folder()
