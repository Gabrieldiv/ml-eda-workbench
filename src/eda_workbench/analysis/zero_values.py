import pandas as pd


def zero_values(df: pd.DataFrame, config: dict) -> str:
    """Report which columns contain zero values, with count and percentage.

    Ignores ID and target columns, as zeros there are valid labels, not data issues.

    Args:
        df: DataFrame with the dataset data.
        config: Dictionary with dataset settings (id_column, target_column).

    Returns:
        Markdown-formatted string with a table of columns containing zero
        values, or a message stating none were found.
    """
    total = len(df)
    id_column = config.get("id_column")
    target_column = config.get("target_column")
    exclude = {col for col in [id_column, target_column] if col is not None}

    df_feat = df.drop(columns=list(exclude), errors="ignore").select_dtypes(
        include="number"
    )
    zero_counts = df_feat.eq(0).sum()
    zero_counts = zero_counts[zero_counts > 0]

    lines = ["## Characteristics with zero values"]

    if zero_counts.empty:
        lines.append("- **No characteristics with zero values**")
        return "\n".join(lines)

    rows = [
        (column, f"{count} instances out of {total}", f"{100 * count / total:.1f}%")
        for column, count in zero_counts.items()
    ]

    headers = ("Characteristic", "Count", "Percentage")
    col1_width = max(len(headers[0]), max(len(r[0]) for r in rows))
    col2_width = max(len(headers[1]), max(len(r[1]) for r in rows))
    col3_width = max(len(headers[2]), max(len(r[2]) for r in rows))

    lines.append(
        f"| {headers[0]:<{col1_width}} | {headers[1]:<{col2_width}} | "
        f"{headers[2]:<{col3_width}} |"
    )
    lines.append(
        f"|{'-' * (col1_width + 2)}|{'-' * (col2_width + 2)}|{'-' * (col3_width + 2)}|"
    )
    for name, count_text, pct_text in rows:
        lines.append(
            f"| {name:<{col1_width}} | {count_text:<{col2_width}} | "
            f"{pct_text:<{col3_width}} |"
        )

    return "\n".join(lines)
