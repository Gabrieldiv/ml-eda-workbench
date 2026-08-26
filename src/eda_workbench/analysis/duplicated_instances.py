import pandas as pd


def duplicated_instances(df: pd.DataFrame, config: dict) -> str:
    """Report groups of instances with identical characteristics.

    Two instances are considered duplicates when every feature value matches,
    ignoring the ID and target columns (the ID is unique by definition, and
    the target is the label we are trying to predict, not a characteristic).

    Args:
        df: DataFrame with the dataset data.
        config: Dictionary with dataset settings (id_column, target_column).

    Returns:
        Markdown-formatted string with a table of duplicated instance groups
        (identified by ID when available, otherwise by row index), including
        a column flagging conflicting target values within a group when
        present. States none were found otherwise. States that no analyzable
        data remains when only ID and target columns are present.
    """
    id_column = config.get("id_column")
    target_column = config.get("target_column")
    exclude = {col for col in [id_column, target_column] if col is not None}

    df_feat = df.drop(columns=list(exclude), errors="ignore")
    lines = ["## Duplicated Instances"]

    if len(df_feat.columns) == 0:
        lines.append(
            "- **After ID and Target columns removal, no analyzable data remains**"
        )
        return "\n".join(lines)

    duplicated_mask = df_feat.duplicated(keep=False)

    if not duplicated_mask.any():
        lines.append("- **No duplicated instances found**")
        return "\n".join(lines)

    feature_columns = list(df_feat.columns)
    groups = df_feat[duplicated_mask].groupby(feature_columns, dropna=False, sort=False)

    rows = []
    for _, group_df in groups:
        row_indices = group_df.index
        if id_column and id_column in df.columns:
            labels = df.loc[row_indices, id_column].astype(str).tolist()
        else:
            labels = [str(i) for i in row_indices]

        conflicting_targets = "—"
        if target_column and target_column in df.columns:
            labels_target = df.loc[row_indices, target_column].astype(str)
            if not (labels_target == labels_target.iloc[0]).all():
                conflicting_targets = ", ".join(labels_target.unique())

        rows.append((str(len(labels)), ", ".join(labels), conflicting_targets))

    headers = ("Group", "Count", "IDs / Row Index", "Conflicting Targets")
    col1_width = max(len(headers[0]), len(str(len(rows))))
    col2_width = max(len(headers[1]), max(len(r[0]) for r in rows))
    col3_width = max(len(headers[2]), max(len(r[1]) for r in rows))
    col4_width = max(len(headers[3]), max(len(r[2]) for r in rows))

    lines.append(
        f"| {headers[0]:<{col1_width}} | {headers[1]:<{col2_width}} | "
        f"{headers[2]:<{col3_width}} | {headers[3]:<{col4_width}} |"
    )
    lines.append(
        f"|{'-' * (col1_width + 2)}|{'-' * (col2_width + 2)}|"
        f"{'-' * (col3_width + 2)}|{'-' * (col4_width + 2)}|"
    )
    for i, (count, ids, conflict) in enumerate(rows, start=1):
        lines.append(
            f"| {str(i):<{col1_width}} | {count:<{col2_width}} | "
            f"{ids:<{col3_width}} | {conflict:<{col4_width}} |"
        )

    return "\n".join(lines)
