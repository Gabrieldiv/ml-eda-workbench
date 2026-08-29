"""Class distribution across one or more label columns."""

import pandas as pd

from eda_workbench.bias.prepare import as_name_list


def class_distribution(
    df: pd.DataFrame, config: dict, high_cardinality_threshold: int = 20
) -> str:
    """Report the frequency distribution of one or more label columns.

    Configured via ``label_columns`` in the dataset YAML. A single
    configured column reports its own value distribution (the standard
    single-label case). Multiple configured columns additionally report
    the most frequent combination of values across them — this is what
    makes the analysis "multilabel": there is no separate flag, the
    behavior is entirely driven by how many columns are configured.

    Missing values (NaN) within a label column are reported as their own
    category, labeled "None" — never inferred or filled in, since doing
    so would misrepresent a data gap as a real, observed label.

    Args:
        df: DataFrame with the dataset data.
        config: Dictionary with dataset settings, including
            ``label_columns`` (a single column name or a list of names).
        high_cardinality_threshold: A label column with more distinct
            values than this is truncated to its most frequent values,
            with a note. Default: 20.

    Returns:
        Markdown-formatted string with a frequency table per configured
        label column, and — when more than one column is configured —
        the most frequent combination of values across them. States when
        no label columns are configured, when configured columns are
        duplicated or missing from the dataset, or when the dataset is
        empty.
    """
    label_columns = as_name_list(config.get("label_columns"))
    lines = ["## Class Distribution"]

    if not label_columns:
        lines.append("- **No label columns configured**")
        return "\n".join(lines)

    duplicates = {c for c in label_columns if label_columns.count(c) > 1}
    if duplicates:
        lines.append(
            "- **Duplicate label column(s) configured**: "
            f"{', '.join(sorted(duplicates))} — fix the configuration "
            "before running this analysis"
        )
        return "\n".join(lines)

    missing_columns = [c for c in label_columns if c not in df.columns]
    present_columns = [c for c in label_columns if c in df.columns]

    if missing_columns:
        lines.append(
            "- **Configured label column(s) not found in dataset**: "
            f"{', '.join(missing_columns)}"
        )

    if not present_columns:
        return "\n".join(lines)

    if len(df) == 0:
        lines.append("- **Empty dataset**")
        return "\n".join(lines)

    total_rows = len(df)

    for column in present_columns:
        series = df[column].fillna("None").astype(str)
        counts = series.value_counts()

        lines.append("")
        lines.append(f"### {column}")

        if len(counts) > high_cardinality_threshold:
            lines.append(
                f"- **High cardinality ({len(counts)} distinct labels) — "
                f"showing top {high_cardinality_threshold}**"
            )
            counts = counts.head(high_cardinality_threshold)

        for value, count in counts.items():
            pct = 100 * count / total_rows
            lines.append(f"- **{value}**: {count} out of {total_rows} ({pct:.1f}%)")

    if len(present_columns) > 1:
        combo_series = df[present_columns].fillna("None").astype(str)
        combo_counts = combo_series.value_counts()
        if not combo_counts.empty:
            top_combo = combo_counts.index[0]
            top_count = combo_counts.iloc[0]
            pct = 100 * top_count / total_rows
            combo_desc = ", ".join(
                f"{column} = {value}"
                for column, value in zip(present_columns, top_combo, strict=True)
            )
            lines.append("")
            lines.append(
                f"**Most common multilabel combination**: {combo_desc} — "
                f"{top_count} out of {total_rows} ({pct:.1f}%)"
            )

    return "\n".join(lines)
