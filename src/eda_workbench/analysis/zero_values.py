import pandas as pd


def zero_values(df: pd.DataFrame, config: dict) -> str:
    """Report which columns contain zero values, with count and percentage.

    Ignores ID and target columns, as zeros there are valid labels, not data issues.

    Args:
        df: DataFrame with the dataset data.
        config: Dictionary with dataset settings (id_column, target_column).

    Returns:
        Formatted string listing columns with zero values, or a message
        stating none were found.
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

    lines = ["Characteristics with zero values:"]

    if zero_counts.empty:
        lines.append("None")
    else:
        for column, count in zero_counts.items():
            pct = 100 * count / total
            lines.append(f"{column} - {count} instances out of {total} - {pct:.1f}%")

    return "\n".join(lines)
