"""Unit tests for pearson_correlation."""

import numpy as np
import pandas as pd

from eda_workbench.analysis import pearson_correlation


def test_skips_non_numeric_columns():
    """Reports non-numeric columns as skipped, without breaking the analysis."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "nome": ["Ana", "Bruno", "Carlos"],
            "idade": [25, 30, 35],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(df, config, limit=0.9)
    assert "**Skipped non-numeric characteristics**" in result
    assert "nome" in result


def test_no_numeric_columns():
    """Reports that no numeric characteristics were found."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "nome": ["Ana", "Bruno", "Carlos"],
            "cargo": ["analista", "gerente", "servente"],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(df, config, limit=0.9)
    assert "**No numeric characteristics found**" in result


def test_finds_redundant_pair():
    """Flags a pair with correlation above the limit and lists a candidate."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a": [1, 2, 3, 4],
            "b": [2, 4, 6, 8],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(df, config, limit=0.9, min_pairwise_n=2)
    assert "a <-> b" in result
    assert "1.000" in result
    assert "Candidate for Review" in result


def test_no_redundant_pairs():
    """Reports that all correlations are below the limit."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a": [1, 5, 2, 8],
            "b": [4, 2, 9, 1],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(df, config, limit=0.9, min_pairwise_n=2)
    assert "**All correlations below the defined limit (0.9)**" in result


def test_handles_constant_column():
    """Flags columns where correlation cannot be computed, without crashing."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a": [1, 2, 3, 4],
            "constante": [5, 5, 5, 5],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(df, config, limit=0.9, min_pairwise_n=2)
    assert "Could not compute correlation (not enough variation/data):" in result
    assert "`a` <-> `constante`" in result


def test_suggests_candidate_with_higher_mean_correlation():
    """Between two redundant columns, lists the one more correlated overall
    as the candidate for review."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "a": [1, 5, 2, 8, 3],
            "b": [1.05, 5.1, 2.1, 8.2, 3.1],
            "c": [1.1, 5.3, 2.2, 8.5, 3.3],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(df, config, limit=0.9, min_pairwise_n=2)
    assert "a <-> b" in result
    assert "b" in result.split("a <-> b")[1].split("\n")[0]


def test_blank_line_separates_skipped_columns_from_table():
    """Adds a blank line between the skipped-columns note and the table
    when both are present, so the table renders correctly."""
    df = pd.DataFrame(
        {
            "id": range(35),
            "nome": ["Ana"] * 35,
            "a": list(range(1, 36)),
            "b": [x * 1.01 for x in range(1, 36)],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(df, config, limit=0.9)

    assert "**Skipped non-numeric characteristics**" in result
    assert "| a <-> b" in result
    assert "**Skipped non-numeric characteristics**: nome\n\n**Candidate" in result


def test_blank_line_separates_skipped_columns_from_nan_pairs():
    """Adds a blank line between the skipped-columns note and the
    'could not compute' section when both are present."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "nome": ["Ana", "Bruno", "Carlos", "Diana"],
            "a": [1, 2, 3, 4],
            "constante": [5, 5, 5, 5],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(df, config, limit=0.9, min_pairwise_n=2)

    assert "**Skipped non-numeric characteristics**" in result
    assert "**Low-cardinality numeric characteristics**" in result
    assert "Could not compute correlation" in result
    assert "distinct values)\n\n**Could not compute correlation" in result


def test_not_enough_numeric_columns():
    """Reports that there are not enough numeric characteristics to compare."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "nome": ["Ana", "Bruno", "Carlos"],
            "a": [10, 20, 30],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(df, config, limit=0.9)
    assert "**Not enough numeric characteristics to compare" in result


def test_drop_suggestion_ignores_uncomputable_pairs():
    """A missing correlation must not count as zero when averaging, since
    that could unfairly bias the candidate recommendation toward the wrong
    column.

    Here, a-b is redundant, a-c is uncomputable (a and c never overlap),
    and b-c is computable but weak. a's only real relationship (with b) is
    perfect, so a should be the candidate — not b, which the old (buggy)
    averaging would have incorrectly picked.
    """
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "a": [1, 2, np.nan, np.nan, np.nan, np.nan],
            "b": [1.01, 2.02, 3.03, 4.04, 5.05, 6.06],
            "c": [np.nan, np.nan, 3, 5, 2, 8],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(df, config, limit=0.9, min_pairwise_n=2)

    assert "`a` <-> `c`" in result
    assert "a <-> b" in result
    assert "a" in result.split("a <-> b")[1].split("\n")[0]


def test_insufficient_pairwise_n_reported_separately():
    """Reports a pair as insufficient data instead of evaluating it against
    the redundancy limit when fewer than min_pairwise_n rows overlap."""
    df = pd.DataFrame(
        {
            "id": range(50),
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [np.nan] * 40,
            "b": [1.1, 2.1, 3.2, 4.1, 5.2, 6.1, 7.2, 8.1, 9.2, 10.1] + [np.nan] * 40,
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(
        df, config, limit=0.9
    )  # default min_pairwise_n=30
    assert "Insufficient overlapping data (fewer than 30" in result
    assert "`a` <-> `b` (N=10 out of 50, 20.0%)" in result
    assert "Candidate for Review" not in result


def test_sufficient_pairwise_n_shown_in_table():
    """Shows N alongside correlation and candidate for pairs with enough data."""
    df = pd.DataFrame(
        {
            "id": range(50),
            "a": list(range(50)),
            "b": [x * 1.01 for x in range(50)],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(df, config, limit=0.9)
    assert "| a <-> b | 1.000       | 50" in result


def test_blank_line_separates_skipped_columns_from_insufficient_n():
    """Adds a blank line between the skipped-columns note and the
    'insufficient overlapping data' section when both are present."""
    df = pd.DataFrame(
        {
            "id": range(50),
            "nome": ["Ana"] * 50,
            "x": list(range(1, 16)) + [np.nan] * 35,
            "y": [v * 1.02 for v in range(1, 16)] + [np.nan] * 35,
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(df, config, limit=0.9)

    assert "**Skipped non-numeric characteristics**" in result
    assert "Insufficient overlapping data" in result
    assert (
        "**Skipped non-numeric characteristics**: nome\n\n"
        "**Insufficient overlapping data" in result
    )


def test_nan_pairs_section_directly_after_header_when_nothing_else_present():
    """No blank line is inserted before the 'could not compute' section
    when nothing else was reported first."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a": [1, 2, 3, 4],
            "constante": [5, 5, 5, 5],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(
        df, config, limit=0.9, min_pairwise_n=2, low_cardinality_threshold=0
    )
    assert "## Pearson Correlation\n**Could not compute correlation" in result


def test_insufficient_n_section_directly_after_header_when_nothing_else_present():
    """No blank line is inserted before the 'insufficient overlapping data'
    section when nothing else was reported first."""
    df = pd.DataFrame(
        {
            "id": range(50),
            "x": list(range(1, 11)) + [np.nan] * 40,
            "y": [v * 1.01 for v in range(1, 11)] + [np.nan] * 40,
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(
        df, config, limit=0.9, low_cardinality_threshold=0
    )
    assert "## Pearson Correlation\n**Insufficient overlapping data" in result


def test_candidate_selection_ignores_insufficient_n_pairs():
    """A pair with insufficient N (correlation near +-1.0 by chance) must
    not inflate a column's average and bias which column is picked as
    candidate for review in an unrelated, trustworthy pair.

    Here, a<->b is the redundant pair under evaluation. Without masking
    insufficient-N pairs before averaging, a<->c (N=2, spuriously ~1.0)
    would inflate a's average enough to make it look more broadly
    correlated than b, incorrectly picking 'a' as the candidate. Once
    a<->c is excluded, b's real second relationship (b<->d) correctly
    makes b the more broadly correlated column instead.
    """
    df = pd.DataFrame(
        {
            "id": range(10),
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [
                np.nan,
                np.nan,
                3.167,
                3.402,
                6.643,
                6.245,
                7.502,
                7.576,
                10.577,
                9.081,
            ],
            "c": [10.0, 90.0] + [np.nan] * 8,
            "d": [
                np.nan,
                np.nan,
                2.552,
                2.604,
                4.567,
                4.698,
                8.586,
                7.788,
                9.714,
                8.176,
            ],
        }
    )
    config = {"id_column": "id", "target_column": None}
    result, _, _ = pearson_correlation(
        df, config, limit=0.85, min_pairwise_n=3, low_cardinality_threshold=0
    )
    assert "`a` <-> `c`" in result
    assert "| a <-> b | 0.929       | 8 out of 10 (80.0%) | b" in result


def test_default_plot_mode_generates_heatmap_only():
    """By default (plot_mode='heatmap'), only the heatmap figure is produced."""
    df = pd.DataFrame({"id": range(10), "a": range(10), "b": range(10)})
    config = {"id_column": "id", "target_column": None}
    _, figures, matrix = pearson_correlation(df, config, limit=0.9, min_pairwise_n=2)
    assert "heatmap" in figures
    assert "violin" not in figures
    assert not matrix.empty


def test_plot_mode_both_generates_both_figures():
    """plot_mode='both' produces both heatmap and violin figures."""
    df = pd.DataFrame({"id": range(10), "a": range(10), "b": range(10)})
    config = {"id_column": "id", "target_column": None}
    _, figures, _ = pearson_correlation(
        df, config, limit=0.9, min_pairwise_n=2, plot_mode="both"
    )
    assert "heatmap" in figures
    assert "violin" in figures


def test_no_figures_or_matrix_when_fewer_than_two_numeric_columns():
    """Returns empty figures dict and empty matrix when there's nothing to plot."""
    df = pd.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
    config = {"id_column": "id", "target_column": None}
    _, figures, matrix = pearson_correlation(df, config, limit=0.9)
    assert figures == {}
    assert matrix.empty


def test_violin_excludes_insufficient_n_pairs_from_data():
    """A characteristic's violin only includes correlations from trustworthy
    (sufficient-N) pairs, not the ones excluded from the text report."""
    df = pd.DataFrame(
        {
            "id": range(10),
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [
                np.nan,
                np.nan,
                3.167,
                3.402,
                6.643,
                6.245,
                7.502,
                7.576,
                10.577,
                9.081,
            ],
            "c": [10.0, 90.0] + [np.nan] * 8,
        }
    )
    config = {"id_column": "id", "target_column": None}
    _, figures, _ = pearson_correlation(
        df,
        config,
        limit=0.85,
        min_pairwise_n=3,
        low_cardinality_threshold=0,
        plot_mode="violin",
    )
    assert "violin" in figures


def test_violin_not_generated_when_no_trustworthy_pairs_exist():
    """When every pair has insufficient N, no characteristic has trustworthy
    correlation data to plot, so no violin figure is generated."""
    df = pd.DataFrame(
        {
            "id": range(10),
            "a": [1, 2] + [np.nan] * 8,
            "b": [3, 4] + [np.nan] * 8,
            "c": [5, 6] + [np.nan] * 8,
        }
    )
    config = {"id_column": "id", "target_column": None}
    _, figures, _ = pearson_correlation(
        df,
        config,
        limit=0.9,
        min_pairwise_n=30,
        low_cardinality_threshold=0,
        plot_mode="violin",
    )
    assert figures == {}
