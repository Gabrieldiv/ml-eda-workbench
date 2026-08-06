import pandas as pd


def outliers_instances(
    df: pd.DataFrame, config: dict, whisker_multiplier: float
) -> str:
    """Report boxplot summary statistics and outliers for each characteristic.

    For each numeric characteristic, computes the five-number summary
    (min, Q1, median, Q3, max) and flags instances whose value falls
    outside the boxplot whisker range: Q1 - whisker_multiplier * IQR to
    Q3 + whisker_multiplier * IQR, where IQR = Q3 - Q1.

    Args:
        df: DataFrame with the dataset data.
        config: Dictionary with dataset settings (id_column, target_column).
        whisker_multiplier: How many IQRs beyond Q1/Q3 define the whisker
            range. The conventional boxplot value is 1.5.

    Returns:
        Markdown-formatted string with a table of summary statistics and
        outlier instances (identified by ID when available, otherwise by
        row index) per numeric characteristic. Lists skipped non-numeric
        characteristics and characteristics with no valid data separately.
    """
    id_column = config.get("id_column")
    target_column = config.get("target_column")
    lines = ["## Characteristic Outliers"]

    exclude = {col for col in [id_column, target_column] if col is not None}
    df_feat = df.drop(columns=list(exclude), errors="ignore")

    numeric_columns = list(df_feat.select_dtypes(include="number").columns)
    skipped_columns = list(df_feat.select_dtypes(exclude="number").columns)

    if whisker_multiplier < 0:
        lines.append(
            f"- **Invalid whisker_multiplier ({whisker_multiplier}): must be >= 0**"
        )
        return "\n".join(lines)

    if len(df) == 0:
        lines.append("- **Empty dataset**")
        return "\n".join(lines)

    if len(df_feat.columns) == 0:
        lines.append(
            "- **After ID and Target columns removal, no analyzable data remains**"
        )
        return "\n".join(lines)

    if skipped_columns:
        lines.append(
            f"- **Skipped non-numeric characteristics**: {', '.join(skipped_columns)}"
        )

    if not numeric_columns:
        lines.append("- **No numeric characteristics found**")
        return "\n".join(lines)

    empty_columns = []
    rows = []

    for column in numeric_columns:
        series = df_feat[column]

        if series.isna().all():
            empty_columns.append(column)
            continue

        stats = series.describe()
        minimum = stats["min"]
        q1 = stats["25%"]
        median = stats["50%"]
        q3 = stats["75%"]
        maximum = stats["max"]

        iqr = q3 - q1
        lower_bound = q1 - whisker_multiplier * iqr
        upper_bound = q3 + whisker_multiplier * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_series = series[outlier_mask]

        if outlier_series.empty:
            outlier_count = "0"
            outlier_text = "—"
        else:
            outlier_count = str(len(outlier_series))

            if id_column and id_column in df.columns:
                id_values = df.loc[outlier_series.index, id_column]
                ids = [
                    f"{row_index} (missing ID)" if pd.isna(id_value) else str(id_value)
                    for row_index, id_value in zip(
                        outlier_series.index, id_values, strict=True
                    )
                ]

            else:
                ids = outlier_series.index.astype(str)

            outlier_items = [
                f"{label}: {value:.2f}"
                for label, value in zip(ids, outlier_series, strict=True)
            ]

            max_displayed_outliers = 10

            if len(outlier_items) > max_displayed_outliers:
                remaining = len(outlier_items) - max_displayed_outliers
                outlier_text = (
                    ", ".join(outlier_items[:max_displayed_outliers])
                    + f", ... and {remaining} more"
                )

            else:
                outlier_text = ", ".join(outlier_items)

        rows.append(
            (
                column,
                f"{minimum:.2f}",
                f"{q1:.2f}",
                f"{median:.2f}",
                f"{q3:.2f}",
                f"{maximum:.2f}",
                outlier_count,
                outlier_text,
            )
        )

    if empty_columns:
        lines.append(
            f"- **Characteristics with no valid data**: {', '.join(empty_columns)}"
        )

    if not rows:
        lines.append("- **No numeric characteristics with valid data**")
        return "\n".join(lines)

    headers = (
        "Characteristic",
        "Min",
        "Q1",
        "Median",
        "Q3",
        "Max",
        "Outliers",
        "Outlier Values (ID: value)",
    )
    widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))
    ]

    lines.append(
        "| "
        + " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
        + " |"
    )
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(v.ljust(w) for v, w in zip(row, widths, strict=True))
            + " |"
        )

    return "\n".join(lines)
