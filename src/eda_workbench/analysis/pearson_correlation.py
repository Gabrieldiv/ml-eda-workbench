import numpy as np
import pandas as pd


def pearson_correlation(df: pd.DataFrame, config: dict, limit: float) -> str:
    """Report pairs of highly correlated (redundant) characteristics.

    Compares every pair of numeric characteristics via Pearson correlation,
    excluding the ID and target columns. Pairs with absolute correlation
    at or above `limit` are flagged as redundant. For each flagged pair,
    suggests which characteristic to drop based on its average correlation with
    every other characteristic — the one that overlaps more with the rest of the
    dataset is suggested for removal.

    Args:
        df: DataFrame with the dataset data.
        config: Dictionary with dataset settings (id_column, target_column).
        limit: Absolute Pearson correlation value (0 to 1) above which two
            characteristics are considered redundant.

    Returns:
        Formatted string listing each redundant pair, their correlation,
        and which characteristics are suggested for removal. Also lists any
        non-numeric characteristics that were skipped.
    """

    id_column = config.get("id_column")
    target_column = config.get("target_column")
    lines = ["Pearson correlation for every characteristic:"]

    exclude = {col for col in [id_column, target_column] if col is not None}
    df_feat = df.drop(columns=list(exclude), errors="ignore")

    numeric_columns = list(df_feat.select_dtypes(include="number").columns)
    skipped_columns = list(df_feat.select_dtypes(exclude="number").columns)

    if skipped_columns:
        lines.append("Skipped non-numeric characteristics:")
        lines.append("\n".join(skipped_columns))

    if numeric_columns:
        df_numeric = df_feat[numeric_columns]
        correlation_matrix = df_numeric.corr()
        labeled_matrix = correlation_matrix.fillna("NOT_COMPUTABLE")
        mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        correlation_pairs_raw = labeled_matrix.where(mask).stack().dropna()  # noqa: PD013

        sentinel = correlation_pairs_raw == "NOT_COMPUTABLE"
        nan_pairs = correlation_pairs_raw[sentinel]

        correlation_pairs = correlation_pairs_raw[~sentinel].astype(float)
        redundant_pairs = correlation_pairs[correlation_pairs.abs() >= limit]

        if nan_pairs.empty and redundant_pairs.empty:
            lines.append(
                f"All characteristics correlations below the defined limit: {limit} "
            )

        else:
            if not nan_pairs.empty:
                lines.append(
                    "Could not compute correlation (not enough variation/data):"
                )
                for column_a, column_b in nan_pairs.index:
                    lines.append(f"{column_a} <-> {column_b}")

            if not redundant_pairs.empty:
                n_columns = len(correlation_matrix)
                mean_abs_corr = (correlation_matrix.abs().sum() - 1) / (n_columns - 1)

                for (column_a, column_b), corr_value in redundant_pairs.items():
                    if mean_abs_corr[column_a] >= mean_abs_corr[column_b]:
                        suggested_drop = column_a
                    else:
                        suggested_drop = column_b

                    lines.append(
                        f"{column_a} <-> {column_b}: {corr_value:.3f} "
                        f"(suggest dropping: {suggested_drop})"
                    )

    else:
        lines.append("Numeric characteristics not found.")

    return "\n".join(lines)
