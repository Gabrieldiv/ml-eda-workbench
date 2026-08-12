import pandas as pd


def basic_info(df: pd.DataFrame, config: dict) -> str:
    """Return basic dataset information formatted as Markdown.

    Args:
        df: DataFrame with the dataset data.
        config: Dictionary with dataset settings (id_column, target_column).

    Returns:
        Markdown-formatted string with number of rows, and number of
        characteristics excluding the ID and target columns when they are
        both configured and actually present in the dataset.
    """
    id_column = config.get("id_column")
    target_column = config.get("target_column")

    id_present = id_column is not None and id_column in df.columns
    target_present = target_column is not None and target_column in df.columns

    exclude = set()
    if id_present:
        exclude.add(id_column)
    if target_present:
        exclude.add(target_column)

    df_feat = df.drop(columns=list(exclude), errors="ignore")
    rows, columns = df_feat.shape

    if id_present:
        id_line = f"- **ID column**: `{id_column}`"
    elif id_column is not None:
        id_line = f"- **ID column `{id_column}` configured but not found in dataset**"
    else:
        id_line = "- **No ID column detected**"

    if target_present:
        target_line = f"- **Target column**: `{target_column}`"
    elif target_column is not None:
        target_line = (
            f"- **Target column `{target_column}` configured but not found in dataset**"
        )
    else:
        target_line = "- **No Target column detected**"

    excluded_labels = []
    if id_present:
        excluded_labels.append("ID")
    if target_present:
        excluded_labels.append("target")

    if excluded_labels:
        columns_label = f"Characteristics (excluding {' and '.join(excluded_labels)})"
    else:
        columns_label = "Characteristics (no ID or target column found in dataset)"

    lines = [
        "## Basic Information",
        id_line,
        target_line,
        f"- **Rows**: {rows}",
        f"- **{columns_label}**: {columns}",
    ]
    return "\n".join(lines)
