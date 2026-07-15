import pandas as pd


def missing_values(df: pd.DataFrame) -> int:
    """Count how many instances have at least one missing value.

    Args:
        df: DataFrame with the dataset data.

    Returns:
        Number of instances with at least one missing value.
    """
    return df.isna().any(axis=1).sum()
