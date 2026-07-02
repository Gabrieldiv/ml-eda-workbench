"""
EDA analysis functions.
"""

import pandas as pd

# -----------------------------------------------------------------------------
# ANALYSIS
# -----------------------------------------------------------------------------


def missing_values(df: pd.DataFrame) -> int:
    """Count how many instances have at least one missing value.

    Args:
        df: DataFrame with the dataset data.

    Returns:
        Number of instances with at least one missing value.
    """
    return df.isna().any(axis=1).sum()


def basic_info(df: pd.DataFrame, id_column: str | None) -> str:
    """Return basic dataset information formatted as text.

    Args:
        df: DataFrame with the dataset data.
        id_column: Name of the ID column if detected, or None if absent.

    Returns:
        Formatted string with number of rows, columns, and ID status.
    """
    rows, columns = df.shape

    if id_column:
        id_line = f"ID column detected: {id_column}"
    else:
        id_line = "No ID column detected."

    lines = [
        "=== BASIC INFORMATION ===",
        id_line,
        f"Rows      : {rows}",
        f"Columns   : {columns}",
    ]
    return "\n".join(lines)


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


def duplicated_instances(df: pd.DataFrame, config: dict) -> str:
    """Report groups of instances with identical characteristics.

    Two instances are considered duplicates when every feature value matches,
    ignoring the ID and target columns (the ID is unique by definition, and
    the target is the label we are trying to predict, not a characteristic).

    Args:
        df: DataFrame with the dataset data.
        config: Dictionary with dataset settings (id_column, target_column).

    Returns:
        Formatted string listing each group of duplicated instances (identified
        by ID when available, otherwise by row position), or a message stating
        none were found.
    """
    id_column = config.get("id_column")
    target_column = config.get("target_column")
    exclude = {col for col in [id_column, target_column] if col is not None}

    df_feat = df.drop(columns=list(exclude), errors="ignore")

    duplicated_mask = df_feat.duplicated(keep=False)
    lines = ["Duplicated instances (identical characteristics):"]

    if not duplicated_mask.any():
        lines.append("None")
        return "\n".join(lines)

    feature_columns = list(df_feat.columns)
    groups = df_feat[duplicated_mask].groupby(feature_columns, dropna=False, sort=False)

    for _, group_df in groups:
        row_indices = group_df.index
        if id_column and id_column in df.columns:
            labels = df.loc[row_indices, id_column].astype(str).tolist()
        else:
            labels = [str(i) for i in row_indices]

        lines.append(f"{len(labels)} instances - IDs/rows: {', '.join(labels)}")

    return "\n".join(lines)
