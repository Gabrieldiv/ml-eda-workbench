import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def pearson_correlation(
    df: pd.DataFrame,
    config: dict,
    limit: float,
    min_pairwise_n: int = 30,
    low_cardinality_threshold: int = 10,
    plot_mode: str = "both",
) -> tuple[str, dict[str, plt.Figure], pd.DataFrame]:
    """Report correlation with the target, and pairs of highly correlated
    (redundant) characteristics.

    First reports, for every numeric characteristic, its Pearson
    correlation with the configured target column (when numeric), sorted
    by absolute strength — descending, regardless of sign. Then compares
    every pair of numeric characteristics against each other, excluding
    the ID and target columns.

    For each correlation computed anywhere in this function — with the
    target, or between characteristics — the number of rows with valid
    (non-null) data in BOTH columns (N) is checked against
    ``min_pairwise_n``: with too few overlapping points, a correlation can
    be misleadingly close to +-1.0 by chance, so such pairs are reported
    separately as insufficient data rather than evaluated normally.

    Among characteristic pairs with enough data, those with absolute
    correlation at or above `limit` are flagged as redundant. For each
    flagged pair, lists a candidate for manual review, based on its
    average correlation with every other characteristic. This is a
    heuristic starting point, not an automatic removal decision: it does
    not account for target association, missingness, domain meaning, or
    validation performance.

    Depending on ``plot_mode``, also builds a heatmap of the full
    characteristic-to-characteristic correlation matrix, a violin plot of
    each characteristic's correlation distribution with every other
    characteristic, or both. In both plots, pairs with insufficient N are
    masked out (not shown), for the same reason they are excluded from
    the "Candidate for Review" heuristic and reported separately in the
    text: an untrustworthy correlation should not visually read as a real
    relationship. These plots cover characteristic-to-characteristic
    correlation only, not correlation with the target.

    Args:
        df: DataFrame with the dataset data.
        config: Dictionary with dataset settings (id_column, target_column).
        limit: Absolute Pearson correlation value (0 to 1) above which two
            characteristics are considered redundant.
        min_pairwise_n: Minimum number of overlapping non-null rows required
            to trust a pairwise correlation. Default: 30.
        low_cardinality_threshold: Numeric characteristics (or the target)
            with this many or fewer distinct values are flagged as
            potentially categorical or ordinal codes, since Pearson
            correlation may not be meaningful for them. Default: 10.
        plot_mode: One of "heatmap", "violin", or "both". Controls which
            correlation plot(s) are generated. Default: "both".

    Raises:
        ValueError: If plot_mode is not one of "heatmap", "violin", or "both".

    Returns:
        A tuple of (markdown_text, figures, correlation_matrix):
        - markdown_text: correlation of each characteristic with the
          target (when the target is numeric), then the table of
          redundant characteristic pairs (including N), their
          correlation, and a candidate for manual review. Also lists
          skipped non-numeric characteristics, low-cardinality numeric
          characteristics, pairs whose correlation could not be computed,
          and pairs (with the target or between characteristics) with
          insufficient overlapping data.
        - figures: dict with keys "heatmap" and/or "violin" mapping to the
          generated Figure objects (characteristic-to-characteristic
          only), per ``plot_mode``. Empty when there are fewer than two
          numeric characteristics to compare.
        - correlation_matrix: the full raw characteristic-to-characteristic
          Pearson correlation matrix (unmasked), for callers that want to
          export it (e.g. as CSV). Empty DataFrame when there are fewer
          than two numeric characteristics to compare.
    """
    id_column = config.get("id_column")
    target_column = config.get("target_column")
    lines = ["## Pearson Correlation"]

    exclude = {col for col in [id_column, target_column] if col is not None}
    df_feat = df.drop(columns=list(exclude), errors="ignore")

    numeric_columns = list(df_feat.select_dtypes(include="number").columns)
    skipped_columns = list(df_feat.select_dtypes(exclude="number").columns)

    valid_plot_modes = {"heatmap", "violin", "both"}
    if plot_mode not in valid_plot_modes:
        raise ValueError(
            f"Invalid plot_mode: {plot_mode!r}. Must be one of "
            f"{', '.join(sorted(valid_plot_modes))}."
        )

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

    # -------------------------------------------------------------------
    # Correlation with the target — reported first, independent of
    # whether there are enough characteristics to compare against each
    # other below.
    # -------------------------------------------------------------------
    if numeric_columns:
        target_lines = _target_correlation_lines(
            df,
            target_column,
            numeric_columns,
            min_pairwise_n,
            low_cardinality_threshold,
        )
        if target_lines:
            lines.append("")
            lines.append("### Correlation with Target")
            lines.extend(target_lines)

    if not numeric_columns:
        lines.append("- **No numeric characteristics found**")
        return "\n".join(lines), {}, pd.DataFrame()

    if len(numeric_columns) < 2:
        lines.append(
            "- **Not enough numeric characteristics to compare (at least two required)**"
        )
        return "\n".join(lines), {}, pd.DataFrame()

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

    # Mask used by BOTH the candidate-selection average and the plots below:
    # hides the diagonal and any pair with insufficient N, so an
    # untrustworthy correlation never counts as a real relationship,
    # whether in a number, a heatmap cell, or a violin's data points.
    no_diagonal = ~np.eye(len(correlation_matrix), dtype=bool)
    sufficient_n = n_matrix >= min_pairwise_n
    trustworthy_mask = (
        pd.DataFrame(
            no_diagonal,
            index=correlation_matrix.index,
            columns=correlation_matrix.columns,
        )
        & sufficient_n
    )

    if nan_pairs.empty and insufficient_n_pairs.empty and redundant_pairs.empty:
        lines.append("")
        lines.append(
            f"- **All pairwise correlations below the defined limit ({limit})**"
        )
        figures = _build_correlation_plots(
            correlation_matrix, trustworthy_mask, plot_mode
        )
        return "\n".join(lines), figures, correlation_matrix

    if not nan_pairs.empty:
        lines.append("")
        lines.append("**Could not compute correlation (not enough variation/data):**")
        lines.append("")
        for column_a, column_b in nan_pairs.index:
            lines.append(f"- `{column_a}` <-> `{column_b}`")

    if not insufficient_n_pairs.empty:
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
        mean_abs_corr = correlation_matrix.abs().where(trustworthy_mask).mean()

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

    figures = _build_correlation_plots(correlation_matrix, trustworthy_mask, plot_mode)
    return "\n".join(lines), figures, correlation_matrix


def _target_correlation_lines(
    df: pd.DataFrame,
    target_column: str | None,
    numeric_columns: list[str],
    min_pairwise_n: int,
    low_cardinality_threshold: int,
) -> list[str]:
    """Build the "Correlation with Target" section, or an empty list when
    there's nothing to report (target not configured).

    Args:
        df: The full, original DataFrame (target_column must still be
            present in it — this is called before target is dropped).
        target_column: Configured target column name, or None.
        numeric_columns: Numeric characteristic columns to correlate
            against the target (already excludes id_column/target_column).
        min_pairwise_n: Same meaning as in ``pearson_correlation``.
        low_cardinality_threshold: Same meaning as in ``pearson_correlation``.

    Returns:
        Markdown lines for this section. Empty when target_column is not
        configured at all.
    """
    if target_column is None:
        return []

    lines: list[str] = []

    if target_column not in df.columns:
        lines.append(
            f"- **Configured target_column not found in dataset**: {target_column}"
        )
        return lines

    target_series = df[target_column]

    if not pd.api.types.is_numeric_dtype(target_series):
        lines.append(
            f"- **Correlation with target not applicable**: `{target_column}` is "
            "not numeric (Pearson correlation requires two numeric variables)"
        )
        return lines

    target_distinct = target_series.nunique()
    if target_distinct <= low_cardinality_threshold:
        lines.append(
            f"- **Target `{target_column}` has low cardinality** "
            f"({target_distinct} distinct values) — may be categorical/ordinal "
            "codes, not a continuous measure"
        )

    total_rows = len(df)
    sufficient_rows: list[tuple[str, float, int]] = []
    insufficient_rows: list[tuple[str, int]] = []

    for column in numeric_columns:
        feature_series = df[column]
        valid_mask = feature_series.notna() & target_series.notna()
        n = int(valid_mask.sum())

        if n < min_pairwise_n:
            insufficient_rows.append((column, n))
            continue

        corr = feature_series[valid_mask].corr(target_series[valid_mask])
        if pd.isna(corr):
            continue
        sufficient_rows.append((column, float(corr), n))

    sufficient_rows.sort(key=lambda row: abs(row[1]), reverse=True)

    if insufficient_rows:
        lines.append("")
        lines.append(
            "**Insufficient overlapping data with target (fewer than "
            f"{min_pairwise_n} valid pairs, correlation not evaluated):**"
        )
        lines.append("")
        for column, n in insufficient_rows:
            pct = 100 * n / total_rows
            lines.append(f"- `{column}` (N={n} out of {total_rows}, {pct:.1f}%)")

    if sufficient_rows:
        headers = ("Characteristic", "Correlation with Target", "N")
        rows_text = [
            (
                column,
                f"{corr:.3f}",
                f"{n} out of {total_rows} ({100 * n / total_rows:.1f}%)",
            )
            for column, corr, n in sufficient_rows
        ]
        col1_width = max(len(headers[0]), max(len(r[0]) for r in rows_text))
        col2_width = max(len(headers[1]), max(len(r[1]) for r in rows_text))
        col3_width = max(len(headers[2]), max(len(r[2]) for r in rows_text))

        lines.append("")
        lines.append(
            f"| {headers[0]:<{col1_width}} | {headers[1]:<{col2_width}} | "
            f"{headers[2]:<{col3_width}} |"
        )
        lines.append(
            f"|{'-' * (col1_width + 2)}|{'-' * (col2_width + 2)}|"
            f"{'-' * (col3_width + 2)}|"
        )
        for column, corr_text, n_text in rows_text:
            lines.append(
                f"| {column:<{col1_width}} | {corr_text:<{col2_width}} | "
                f"{n_text:<{col3_width}} |"
            )

    return lines


def _build_correlation_plots(
    correlation_matrix: pd.DataFrame,
    trustworthy_mask: pd.DataFrame,
    plot_mode: str,
) -> dict[str, plt.Figure]:
    """Build the requested correlation plot(s), masking untrustworthy pairs.

    Args:
        correlation_matrix: Full raw Pearson correlation matrix.
        trustworthy_mask: True where a cell should be shown (not the
            diagonal, and N at or above the configured minimum).
        plot_mode: One of "heatmap", "violin", or "both".

    Returns:
        Dict with keys "heatmap" and/or "violin" mapping to Figure objects,
        for whichever plot(s) ``plot_mode`` requests.
    """
    figures: dict[str, plt.Figure] = {}
    hide_mask = ~trustworthy_mask

    if plot_mode in ("heatmap", "both"):
        fig, ax = plt.subplots(
            figsize=(
                max(6, len(correlation_matrix) * 0.8),
                max(5, len(correlation_matrix) * 0.7),
            )
        )
        sns.heatmap(
            correlation_matrix,
            mask=hide_mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            ax=ax,
            cbar_kws={"label": "Pearson correlation"},
        )
        ax.set_title("Pearson Correlation Heatmap (blank: insufficient N or diagonal)")
        fig.tight_layout()
        figures["heatmap"] = fig

    if plot_mode in ("violin", "both"):
        trustworthy_values = correlation_matrix.where(trustworthy_mask)
        columns = list(correlation_matrix.columns)

        plotted_columns = []
        dataset = []
        for column in columns:
            values = trustworthy_values[column].dropna().to_numpy()
            if len(values) > 0:
                plotted_columns.append(column)
                dataset.append(values)

        if dataset:
            fig, ax = plt.subplots(figsize=(max(6, len(plotted_columns) * 0.8), 6))
            ax.violinplot(dataset, showmeans=True, showextrema=True)
            ax.set_xticks(range(1, len(plotted_columns) + 1))
            ax.set_xticklabels(plotted_columns, rotation=45, ha="right")
            ax.set_ylabel("Pearson correlation with other characteristics")
            ax.set_title(
                "Distribution of Pairwise Correlations per Characteristic "
                "(insufficient-N pairs excluded)"
            )
            ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
            fig.tight_layout()
            figures["violin"] = fig

    return figures
