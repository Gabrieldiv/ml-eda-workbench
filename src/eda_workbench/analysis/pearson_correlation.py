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
        Markdown-formatted string with a table of redundant characteristic
        pairs, their correlation, and which one is suggested for removal.
        Also lists skipped non-numeric characteristics and pairs whose
        correlation could not be computed.
    """
    id_column = config.get("id_column")
    target_column = config.get("target_column")
    lines = ["## Pearson Correlation"]

    exclude = {col for col in [id_column, target_column] if col is not None}
    df_feat = df.drop(columns=list(exclude), errors="ignore")

    numeric_columns = list(df_feat.select_dtypes(include="number").columns)
    skipped_columns = list(df_feat.select_dtypes(exclude="number").columns)

    if skipped_columns:
        lines.append(
            f"- **Skipped non-numeric characteristics**: {', '.join(skipped_columns)}"
        )

    if not numeric_columns:
        lines.append("- **No numeric characteristics found**")
        return "\n".join(lines)

    if len(numeric_columns) < 2:
        lines.append(
            "- **Not enough numeric characteristics to compare (at least two required)**"
        )
        return "\n".join(lines)

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
        lines.append(f"- **All correlations below the defined limit ({limit})**")
        return "\n".join(lines)

    if not nan_pairs.empty:
        if len(lines) > 1:
            lines.append("")

        lines.append("**Could not compute correlation (not enough variation/data):**")
        lines.append("")
        for column_a, column_b in nan_pairs.index:
            lines.append(f"- `{column_a}` <-> `{column_b}`")

    if not redundant_pairs.empty:
        no_diagonal = ~np.eye(len(correlation_matrix), dtype=bool)
        mean_abs_corr = correlation_matrix.abs().where(no_diagonal).mean()

        rows = []
        for (column_a, column_b), corr_value in redundant_pairs.items():
            if mean_abs_corr[column_a] >= mean_abs_corr[column_b]:
                suggested_drop = column_a
            else:
                suggested_drop = column_b

            rows.append(
                (f"{column_a} <-> {column_b}", f"{corr_value:.3f}", suggested_drop)
            )

        headers = ("Pair", "Correlation", "Suggested Drop")
        col1_width = max(len(headers[0]), max(len(r[0]) for r in rows))
        col2_width = max(len(headers[1]), max(len(r[1]) for r in rows))
        col3_width = max(len(headers[2]), max(len(r[2]) for r in rows))

        if len(lines) > 1:
            lines.append("")

        lines.append(
            f"| {headers[0]:<{col1_width}} | {headers[1]:<{col2_width}} | "
            f"{headers[2]:<{col3_width}} |"
        )
        lines.append(
            f"|{'-' * (col1_width + 2)}|{'-' * (col2_width + 2)}|"
            f"{'-' * (col3_width + 2)}|"
        )
        for pair, corr_text, drop in rows:
            lines.append(
                f"| {pair:<{col1_width}} | {corr_text:<{col2_width}} | "
                f"{drop:<{col3_width}} |"
            )

    return "\n".join(lines)
