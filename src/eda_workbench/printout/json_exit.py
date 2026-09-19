"""JSON renderers for structured analysis results."""

from dataclasses import asdict

from eda_workbench.analysis.types import MissingValuesResult


def render_missing_values_json(result: MissingValuesResult) -> dict:
    """Convert a MissingValuesResult into a JSON-serializable dict.

    Args:
        result: Structured result from missing_values().

    Returns:
        Dict with the same fields as the dataclass, ready to be nested
        inside a larger report dict and serialized with json.dumps().
    """
    return asdict(result)
