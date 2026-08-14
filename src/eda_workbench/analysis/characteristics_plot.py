import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def characteristics_plot(
    df: pd.DataFrame, config: dict
) -> tuple[str, plt.Figure | None]:
    """Plot standardized min, max, mean, and mean±std for each numeric characteristic.

    Excludes the ID and target columns. Each numeric value is standardized
    (z-score: (value - mean) / std) before plotting, so characteristics with
    very different raw scales and units become directly comparable on a
    single shared y-axis. After standardization, the mean always maps to 0
    and mean±std always maps to ±1 for every characteristic; only min and
    max vary meaningfully between characteristics. Precise raw values (min,
    Q1, median, Q3, max) remain available in the outliers report table.

    Args:
        df: DataFrame with the dataset data.
        config: Dictionary with dataset settings (id_column, target_column).

    Returns:
        A tuple of (markdown_text, figure). figure is None when there are
        no numeric characteristics with valid, non-constant data to plot.
    """
    id_column = config.get("id_column")
    target_column = config.get("target_column")
    lines = ["## Characteristics Value Ranges Graphic"]

    exclude = {col for col in [id_column, target_column] if col is not None}
    df_feat = df.drop(columns=list(exclude), errors="ignore")

    numeric_columns = list(df_feat.select_dtypes(include="number").columns)
    skipped_columns = list(df_feat.select_dtypes(exclude="number").columns)

    if len(df) == 0:
        lines.append("- **Empty dataset**")
        return "\n".join(lines), None

    if len(df_feat.columns) == 0:
        lines.append(
            "- **After ID and Target columns removal, no analyzable data remains**"
        )
        return "\n".join(lines), None

    if not numeric_columns:
        lines.append("- **No numeric characteristics found**")
        return "\n".join(lines), None

    if skipped_columns:
        lines.append(
            f"- **Skipped non-numeric characteristics**: {', '.join(skipped_columns)}"
        )

    empty_columns = []
    constant_columns = []
    valid_columns, z_minimums, z_maximums = [], [], []

    for column in numeric_columns:
        series = df_feat[column]

        if series.isna().all():
            empty_columns.append(column)
            continue

        mean = series.mean()
        std = series.std()

        if std == 0:
            constant_columns.append(column)
            continue

        valid_columns.append(column)
        z_minimums.append((series.min() - mean) / std)
        z_maximums.append((series.max() - mean) / std)

    if empty_columns:
        lines.append(
            f"- **Characteristics with no valid data**: {', '.join(empty_columns)}"
        )

    if constant_columns:
        lines.append(
            "- **Constant characteristics (no variability, cannot standardize)**: "
            f"{', '.join(constant_columns)}"
        )

    if not valid_columns:
        lines.append("- **No numeric characteristics with valid data**")
        return "\n".join(lines), None

    x = range(len(valid_columns))
    z_means = [0.0] * len(valid_columns)
    z_lower_std = [-1.0] * len(valid_columns)
    z_upper_std = [1.0] * len(valid_columns)

    fig, ax = plt.subplots(figsize=(max(6, len(valid_columns) * 0.8), 6))
    ax.scatter(x, z_minimums, color="#1f77b4", label="Min (z-score)")
    ax.scatter(x, z_maximums, color="#9467bd", label="Max (z-score)")
    ax.scatter(x, z_means, color="#2ca02c", label="Mean (z-score)")
    ax.scatter(x, z_lower_std, color="#ff7f0e", label="Mean ± Std (z-score)")
    ax.scatter(x, z_upper_std, color="#ff7f0e")

    ax.set_xticks(list(x))
    ax.set_xticklabels(valid_columns, rotation=45, ha="right")
    ax.set_xlabel("Characteristic")
    ax.set_ylabel("Standardized Value (z-score)")
    ax.set_title("Standardized Characteristic Value Ranges (Min / Mean ± Std / Max)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return "\n".join(lines), fig
