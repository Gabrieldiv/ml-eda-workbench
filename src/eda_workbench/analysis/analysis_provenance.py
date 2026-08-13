"""Analysis provenance metadata for reproducibility."""

from datetime import datetime
from importlib.metadata import version
from zoneinfo import ZoneInfo


def analysis_provenance(
    dataset_name: str,
    limit: float,
    min_pairwise_n: int,
    whisker_multiplier: float,
    low_cardinality_threshold: int,
) -> str:
    """Report the configuration and environment used to generate this analysis.

    Records dataset identity, every threshold applied by the other analysis
    functions, the versions of the key packages involved, and the
    generation timestamp — so a report can be reproduced or audited later.
    The configured ID/target columns are reported by ``basic_info``, right
    after this section, to avoid duplication.

    Args:
        dataset_name: Name of the dataset file (e.g. "wine.csv"), without
            the containing folder path, so provenance stays portable
            across machines with different directory layouts.
        limit: The Pearson correlation redundancy threshold in use.
        min_pairwise_n: The minimum pairwise sample size threshold in use.
        whisker_multiplier: The boxplot whisker multiplier in use.
        low_cardinality_threshold: The low-cardinality warning threshold
            in use.

    Returns:
        Markdown-formatted string listing the analysis provenance.
    """
    generated_at = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime(
        "%Y-%m-%d %H:%M:%S %z (America/Sao_Paulo)"
    )

    lines = [
        "## Analysis Provenance",
        f"- **Dataset**: {dataset_name}",
        f"- **Correlation redundancy threshold**: {limit}",
        f"- **Minimum pairwise N for correlation**: {min_pairwise_n}",
        f"- **Outlier whisker multiplier**: {whisker_multiplier}",
        f"- **Low-cardinality threshold**: {low_cardinality_threshold}",
        f"- **Package versions**: pandas {version('pandas')}, "
        f"numpy {version('numpy')}, matplotlib {version('matplotlib')}",
        f"- **Generated at**: {generated_at}",
    ]

    return "\n".join(lines)
