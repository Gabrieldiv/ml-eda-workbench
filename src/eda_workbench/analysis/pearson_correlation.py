import pandas as pd


def pearson_correlation(df: pd.DataFrame, config: dict) -> str:
    id_column = config.get("id_column")
    target_column = config.get("target_column")
    lines = ["Pearson correlation for every characteristic:"]

    if (
        not target_column
        or target_column not in df.columns
        or not pd.api.types.is_numeric_dtype(df[target_column])
    ):
        lines.append(
            "Pearson correlation not viable, target isn't numeric or doesn't exist."
        )
        return "\n".join(lines)

    exclude = {col for col in [id_column, target_column] if col is not None}
    df_feat = df.drop(columns=list(exclude), errors="ignore")

    numeric_features = set(df_feat.select_dtypes(include="number").columns)
    skipped_features = set(df_feat.select_dtypes(exclude="number").columns)

    if skipped_features:
        lines.append(
            f"Skipped non-numeric columns: {', '.join(sorted(skipped_features))}"
        )

    if numeric_features:
        df_numeric = df_feat[list(numeric_features)]
        correlations = df_numeric.corrwith(df[target_column])
        correlations = correlations.reindex(
            correlations.abs().sort_values(ascending=False).index
        )
        for feature_name, corr_value in correlations.items():
            lines.append(f"{feature_name}: {corr_value:.3f}")

    return "\n".join(lines)
