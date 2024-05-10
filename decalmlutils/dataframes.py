import pandas as pd
from beartype import beartype


@beartype
def from_multihot(df: pd.DataFrame) -> pd.Series:
    """
    Given a multihot df, return a df where each item is a list of labels.
    """

    def agg_row(row):
        return [class_ for class_, val in row.items() if val == 1]

    return df.apply(lambda row: agg_row(row), axis=1)
