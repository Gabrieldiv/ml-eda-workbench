import numpy as np
import pandas as pd


def redundant_features(df: pd.DataFrame, config: dict, threshold: float) -> str:
    """..."""
    id_column = config.get("id_column")
    target_column = config.get("target_column")
    lines = ["Redundant feature pairs (Pearson correlation):"]

    exclude = {col for col in [id_column, target_column] if col is not None}
    df_feat = df.drop(columns=list(exclude), errors="ignore")

    numeric_features = set(df_feat.select_dtypes(include="number").columns)
    skipped_features = set(df_feat.select_dtypes(exclude="number").columns)

    if skipped_features:
        lines.append(
            f"Skipped non-numeric columns: {', '.join(sorted(skipped_features))}"
        )

    if not numeric_features:
        lines.append("None")
        return "\n".join(lines)

    df_numeric = df_feat[list(numeric_features)]
    correlation_matrix = df_numeric.corr()
    mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    correlation_pairs = correlation_matrix.where(mask).stack().dropna()
    redundant_pairs = correlation_pairs[correlation_pairs.abs() >= threshold]

    if redundant_pairs.empty:
        lines.append("None")
        return "\n".join(lines)

    n_features = len(correlation_matrix)
    mean_abs_corr = (correlation_matrix.abs().sum() - 1) / (n_features - 1)

    for (feature_a, feature_b), corr_value in redundant_pairs.items():
        if mean_abs_corr[feature_a] >= mean_abs_corr[feature_b]:
            suggested_drop = feature_a
        else:
            suggested_drop = feature_b

        lines.append(
            f"{feature_a} <-> {feature_b}: {corr_value:.3f} "
            f"(suggest dropping: {suggested_drop})"
        )

    return "\n".join(lines)