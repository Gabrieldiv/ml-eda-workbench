import numpy as np
import pandas as pd


def pearson_correlation(
    df: pd.DataFrame,
    config: dict,
    limit: float,
    min_pairwise_n: int = 30,
    low_cardinality_threshold: int = 10,
) -> str:
    """Report pairs of highly correlated (redundant) characteristics.

    Compares every pair of numeric characteristics via Pearson correlation,
    excluding the ID and target columns. For each pair, the number of rows
    with valid (non-null) data in BOTH columns (N) is checked against
    ``min_pairwise_n``: with too few overlapping points, a correlation can
    be misleadingly close to +-1.0 by chance, so such pairs are reported
    separately as insufficient data rather than evaluated against `limit`.

    Among pairs with enough data, those with absolute correlation at or
    above `limit` are flagged as redundant. For each flagged pair, lists a
    candidate for manual review, based on its average correlation with
    every other characteristic. This is a heuristic starting point, not an
    automatic removal decision: it does not account for target association,
    missingness, domain meaning, or validation performance.

    Args:
        df: DataFrame with the dataset data.
        config: Dictionary with dataset settings (id_column, target_column).
        limit: Absolute Pearson correlation value (0 to 1) above which two
            characteristics are considered redundant.
        min_pairwise_n: Minimum number of overlapping non-null rows required
            to trust a pairwise correlation. Default: 30.
        low_cardinality_threshold: Numeric characteristics with this many or
            fewer distinct values are flagged as potentially categorical or
            ordinal codes, since Pearson correlation may not be meaningful
            for them. Default: 10.

    Returns:
        Markdown-formatted string with a table of redundant characteristic
        pairs (including N), their correlation, and a candidate for manual
        review. Also lists skipped non-numeric characteristics, low-
        cardinality numeric characteristics, pairs whose correlation could
        not be computed, and pairs with insufficient overlapping data.
    """
    id_column = config.get("id_column")
    target_column = config.get("target_column")
    lines = ["## Pearson Correlation"]

    exclude = {col for col in [id_column, target_column] if col is not None}
    df_feat = df.drop(columns=list(exclude), errors="ignore")

    numeric_columns = list(df_feat.select_dtypes(include="number").columns)
    skipped_columns = list(df_feat.select_dtypes(exclude="number").columns)

    low_cardinality_columns = [
        f"{column} ({df_feat[column].nunique()} distinct values)"
        for column in numeric_columns
        if df_feat[column].nunique() <= low_cardinality_threshold
    ]

    if skipped_columns:
        lines.append(
            f"- **Skipped non-numeric characteristics**: {', '.join(skipped_columns)}"
        )

    if low_cardinality_columns:
        lines.append(
            "- **Low-cardinality numeric characteristics** (may be categorical/ordinal "
            f"codes, not continuous measures): {', '.join(low_cardinality_columns)}"
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
    total_rows = len(df_numeric)
    correlation_matrix = df_numeric.corr()

    notna = df_numeric.notna().astype(int)
    n_matrix = notna.T.dot(notna)

    labeled_matrix = correlation_matrix.fillna("NOT_COMPUTABLE")
    mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    correlation_pairs_raw = labeled_matrix.where(mask).stack().dropna()  # noqa: PD013
    n_pairs = n_matrix.where(mask).stack()  # noqa: PD013

    sentinel = correlation_pairs_raw == "NOT_COMPUTABLE"
    nan_pairs = correlation_pairs_raw[sentinel]

    correlation_pairs = correlation_pairs_raw[~sentinel].astype(float)
    n_for_valid_pairs = n_pairs.reindex(correlation_pairs.index)

    insufficient_mask = n_for_valid_pairs < min_pairwise_n
    insufficient_n_pairs = correlation_pairs[insufficient_mask]
    insufficient_n_values = n_for_valid_pairs[insufficient_mask]

    sufficient_pairs = correlation_pairs[~insufficient_mask]
    redundant_pairs = sufficient_pairs[sufficient_pairs.abs() >= limit]
    redundant_n_values = n_for_valid_pairs[~insufficient_mask][
        sufficient_pairs.abs() >= limit
    ]

    if nan_pairs.empty and insufficient_n_pairs.empty and redundant_pairs.empty:
        lines.append(f"- **All correlations below the defined limit ({limit})**")
        return "\n".join(lines)

    if not nan_pairs.empty:
        if len(lines) > 1:
            lines.append("")

        lines.append("**Could not compute correlation (not enough variation/data):**")
        lines.append("")
        for column_a, column_b in nan_pairs.index:
            lines.append(f"- `{column_a}` <-> `{column_b}`")

    if not insufficient_n_pairs.empty:
        if len(lines) > 1:
            lines.append("")

        lines.append(
            f"**Insufficient overlapping data (fewer than {min_pairwise_n} "
            "valid pairs, correlation not evaluated):**"
        )
        lines.append("")
        for (column_a, column_b), n_value in insufficient_n_values.items():
            n_int = int(n_value)
            pct = 100 * n_int / total_rows
            lines.append(
                f"- `{column_a}` <-> `{column_b}` "
                f"(N={n_int} out of {total_rows}, {pct:.1f}%)"
            )

    if not redundant_pairs.empty:
        no_diagonal = ~np.eye(len(correlation_matrix), dtype=bool)
        sufficient_n = n_matrix >= min_pairwise_n
        combined_mask = (
            pd.DataFrame(
                no_diagonal,
                index=correlation_matrix.index,
                columns=correlation_matrix.columns,
            )
            & sufficient_n
        )
        mean_abs_corr = correlation_matrix.abs().where(combined_mask).mean()

        rows = []
        for (column_a, column_b), corr_value in redundant_pairs.items():
            n_value = redundant_n_values[(column_a, column_b)]
            if mean_abs_corr[column_a] >= mean_abs_corr[column_b]:
                candidate = column_a
            else:
                candidate = column_b

            n_int = int(n_value)
            pct = 100 * n_int / total_rows
            rows.append(
                (
                    f"{column_a} <-> {column_b}",
                    f"{corr_value:.3f}",
                    f"{n_int} out of {total_rows} ({pct:.1f}%)",
                    candidate,
                )
            )

        headers = ("Pair", "Correlation", "N", "Candidate for Review")
        col1_width = max(len(headers[0]), max(len(r[0]) for r in rows))
        col2_width = max(len(headers[1]), max(len(r[1]) for r in rows))
        col3_width = max(len(headers[2]), max(len(r[2]) for r in rows))
        col4_width = max(len(headers[3]), max(len(r[3]) for r in rows))

        if len(lines) > 1:
            lines.append("")

        lines.append(
            "**Candidate for Review**: based on average correlation only, "
            "not an automatic removal decision."
        )
        lines.append("")

        lines.append(
            f"| {headers[0]:<{col1_width}} | {headers[1]:<{col2_width}} | "
            f"{headers[2]:<{col3_width}} | {headers[3]:<{col4_width}} |"
        )
        lines.append(
            f"|{'-' * (col1_width + 2)}|{'-' * (col2_width + 2)}|"
            f"{'-' * (col3_width + 2)}|{'-' * (col4_width + 2)}|"
        )
        for pair, corr_text, n_text, candidate in rows:
            lines.append(
                f"| {pair:<{col1_width}} | {corr_text:<{col2_width}} | "
                f"{n_text:<{col3_width}} | {candidate:<{col4_width}} |"
            )

    return "\n".join(lines)
