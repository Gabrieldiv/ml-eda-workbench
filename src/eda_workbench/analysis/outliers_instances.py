import pandas as pd


def outliers_instances(
    df: pd.DataFrame,
    config: dict,
    whisker_multiplier: float,
    low_cardinality_threshold: int = 10,
) -> str:
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
    low_cardinality_columns = []
    rows = []
    max_displayed_items = 8

    for column in numeric_columns:
        series = df_feat[column]

        if series.isna().all():
            empty_columns.append(column)
            continue

        valid_count = int(series.notna().sum())
        nunique = series.nunique()
        if nunique <= low_cardinality_threshold:
            low_cardinality_columns.append(f"{column} ({nunique} distinct values)")

        stats = series.describe()
        minimum = stats["min"]
        q1 = stats["25%"]
        median = stats["50%"]
        q3 = stats["75%"]
        maximum = stats["max"]

        iqr = q3 - q1

        if iqr == 0:
            value_counts = series.dropna().value_counts()
            total_valid = value_counts.sum()
            freq_items = [
                f"{value}: {count} ({100 * count / total_valid:.1f}%)"
                for value, count in value_counts.items()
            ]

            if len(freq_items) > max_displayed_items:
                remaining = len(freq_items) - max_displayed_items
                outlier_text = (
                    ", ".join(freq_items[:max_displayed_items])
                    + f", ... and {remaining} more"
                )
            else:
                outlier_text = ", ".join(freq_items)

            outlier_count = "N/A (zero IQR)"

        else:
            lower_bound = q1 - whisker_multiplier * iqr
            upper_bound = q3 + whisker_multiplier * iqr

            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_series = series[outlier_mask]

            if outlier_series.empty:
                outlier_count = f"0 out of {valid_count} (0.0%)"
                outlier_text = "—"
            else:
                n_outliers = len(outlier_series)
                outlier_pct = 100 * n_outliers / valid_count
                outlier_count = (
                    f"{n_outliers} out of {valid_count} ({outlier_pct:.1f}%)"
                )

                if id_column and id_column in df.columns:
                    id_values = df.loc[outlier_series.index, id_column]
                    ids = [
                        f"{row_index} (missing ID)"
                        if pd.isna(id_value)
                        else str(id_value)
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

                if len(outlier_items) > max_displayed_items:
                    remaining = len(outlier_items) - max_displayed_items
                    outlier_text = (
                        ", ".join(outlier_items[:max_displayed_items])
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

    if low_cardinality_columns:
        lines.append(
            "- **Low-cardinality numeric characteristics** (may be categorical/ordinal "
            f"codes, not continuous measures): {', '.join(low_cardinality_columns)}"
        )

    if empty_columns:
        lines.append(
            f"- **Characteristics with no valid data**: {', '.join(empty_columns)}"
        )

    if not rows:
        lines.append("- **No numeric characteristics with valid data**")
        return "\n".join(lines)

    if len(lines) > 1:
        lines.append("")

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
