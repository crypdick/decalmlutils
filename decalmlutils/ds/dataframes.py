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


def pandas_print_everything():
    """
    Sets global pandas print options to print everything.
    """
    import pandas as pd

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
