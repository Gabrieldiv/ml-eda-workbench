import pandas as pd


def basic_info(df: pd.DataFrame, id_column: str | None) -> str:
    """Return basic dataset information formatted as text.

    Args:
        df: DataFrame with the dataset data.
        id_column: Name of the ID column if detected, or None if absent.

    Returns:
        Formatted string with number of rows, columns, and ID status.
    """
    rows, columns = df.shape

    if id_column:
        id_line = f"ID column detected: {id_column}"
    else:
        id_line = "No ID column detected."

    lines = [
        "=== BASIC INFORMATION ===",
        id_line,
        f"Rows      : {rows}",
        f"Columns   : {columns}",
    ]
    return "\n".join(lines)
