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
        Formatted string listing each group of duplicated instances (identified
        by ID when available, otherwise by row index), or a message stating
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

        lines.append(f"{len(labels)} instances - IDs/row index: {', '.join(labels)}")

        if target_column and target_column in df.columns:
            labels_target = df.loc[row_indices, target_column].astype(str)

            if not (labels_target == labels_target.iloc[0]).all():
                lines.append(
                    f"{len(labels_target)} instances - different targets: "
                    f"{', '.join(labels_target.unique())}"
                )

    return "\n".join(lines)
