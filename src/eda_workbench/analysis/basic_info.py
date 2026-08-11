import pandas as pd


def basic_info(df: pd.DataFrame, config: dict) -> str:
    """Return basic dataset information formatted as Markdown.

    Args:
        df: DataFrame with the dataset data.
        config: Dictionary with dataset settings (id_column, target_column).

    Returns:
        Markdown-formatted string with number of rows, and number of
        characteristics excluding the ID and target columns when configured.
    """
    id_column = config.get("id_column")
    target_column = config.get("target_column")

    exclude = {col for col in [id_column, target_column] if col is not None}
    df_feat = df.drop(columns=list(exclude), errors="ignore")

    rows, columns = df_feat.shape

    if id_column:
        id_line = f"- **ID column**: `{id_column}`"
    else:
        id_line = "- **No ID column detected**"

    if target_column:
        target_line = f"- **Target column**: `{target_column}`"
    else:
        target_line = "- **No Target column detected**"

    excluded_labels = []
    if id_column:
        excluded_labels.append("ID")
    if target_column:
        excluded_labels.append("target")

    if excluded_labels:
        columns_label = f"Characteristics (excluding {' and '.join(excluded_labels)})"
    else:
        columns_label = (
            "Characteristics (ID and target columns not configured in settings.yaml)"
        )

    lines = [
        "## Basic Information",
        id_line,
        target_line,
        f"- **Rows**: {rows}",
        f"- **{columns_label}**: {columns}",
    ]
    return "\n".join(lines)
