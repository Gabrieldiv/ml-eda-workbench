import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def characteristics_plot(
    df: pd.DataFrame, config: dict
) -> tuple[str, plt.Figure | None]:
    """Plot min, max, mean, and mean±std for each numeric characteristic.

    Excludes the ID and target columns. Each characteristic appears on the
    x-axis; its minimum, mean minus one standard deviation, mean, mean plus
    one standard deviation, and maximum are plotted as differently colored
    points on the y-axis.

    Args:
        df: DataFrame with the dataset data.
        config: Dictionary with dataset settings (id_column, target_column).

    Returns:
        A tuple of (markdown_text, figure). figure is None when there are
        no numeric characteristics with valid data to plot.
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
    valid_columns, minimums, maximums, means, standard_deviations = [], [], [], [], []

    for column in numeric_columns:
        series = df_feat[column]

        if series.isna().all():
            empty_columns.append(column)
            continue

        stats = series.describe()
        valid_columns.append(column)
        minimums.append(stats["min"])
        maximums.append(stats["max"])
        means.append(stats["mean"])
        standard_deviations.append(stats["std"])

    if empty_columns:
        lines.append(
            f"- **Characteristics with no valid data**: {', '.join(empty_columns)}"
        )

    if not valid_columns:
        lines.append("- **No numeric characteristics with valid data**")
        return "\n".join(lines), None

    x = range(len(valid_columns))
    lower_std = [m - s for m, s in zip(means, standard_deviations, strict=True)]
    upper_std = [m + s for m, s in zip(means, standard_deviations, strict=True)]

    fig, ax = plt.subplots(figsize=(max(6, len(valid_columns) * 0.8), 6))
    ax.scatter(x, minimums, color="#1f77b4", label="Min")
    ax.scatter(x, maximums, color="#9467bd", label="Max")
    ax.scatter(x, means, color="#2ca02c", label="Mean")
    ax.scatter(x, lower_std, color="#ff7f0e", label="Mean ± Std")
    ax.scatter(x, upper_std, color="#ff7f0e")

    ax.set_xticks(list(x))
    ax.set_xticklabels(valid_columns, rotation=45, ha="right")
    ax.set_xlabel("Characteristic")
    ax.set_ylabel("Value")
    ax.set_title("Characteristic Value Ranges (Min / Mean ± Std / Max)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return "\n".join(lines), fig
