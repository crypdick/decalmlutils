import logging

import pandas as pd
from beartype import beartype
from beartype.typing import Any, Dict, Iterable, List, Union
from natsort import natsorted
from pandas.core.indexes.base import Index

logger = logging.getLogger(
    __name__
)  # do not use cloudwatch logging here due to circular imports


@beartype
def natural_sort(
    iterable: Iterable, reverse: bool = False, key=None, warnings: bool = True
) -> Union[List, Dict[Union[str, int], Any]]:
    """
    Sort the given list, dict, Pandas Series/Index or collections.Counter in the way that humans expect.
    """
    if isinstance(iterable, (Index, pd.Series)):
        iterable = iterable.tolist()
        if warnings:
            logger.warning(
                "Function natural_sort converts pandas Indexes/Series into lists!"
            )
    elif isinstance(iterable, dict):  # included Counters
        # dicts require special treatment
        return dict(natsorted(iterable.items(), reverse=reverse, key=key))
    elif not isinstance(iterable, list):
        # all other iterables are coerced into list first
        if warnings:
            logger.warning("Function natural_sort converts iterables into lists!")
        iterable = list(iterable)

    return natsorted(iterable, reverse=reverse, key=key)
