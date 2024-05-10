import hashlib
import math
import pprint
import random
import re
import string
from datetime import datetime, timezone

import numpy as np
from beartype import beartype
from beartype.typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from mltoolkit.io.sort import natural_sort

DateType = Literal[None, "date", "datetime"]


@beartype
def millify(n: Union[int, float], num_decimals: int = 0) -> str:
    """
    Converts a large number into "metric equivalents", e.g. 1000 --> 1 K.

    Useful for plots which have limited real estate.

    Limitations: does not work with infinity values, NaN
    """
    assert 0 <= num_decimals < 3
    n = float(n)
    assert np.isfinite(n)

    if abs(n) < 1000:
        millified = f"{n:.{num_decimals}f}"
    else:
        millnames = ["", " K", " M", " B", " T"]
        millidx = max(
            0,
            min(
                len(millnames) - 1,
                int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
            ),
        )
        millified = ("{:." + str(num_decimals) + "f}{}").format(
            n / 10 ** (3 * millidx), millnames[millidx]
        )

    return millified


@beartype
def flatten_list(list_: Union[List, Set, Tuple]) -> List:
    """
    Flatten an irregular list of lists (or sets, tuples) into a single unnested list.

    Will not flatten numpy arrays, which is useful for when we need to flatten inference outputs


    Args:
        list_: a list that may contain arbitrarily nested lists

    Returns:
        a flat list
    """

    def flatten_generator(irregular_list):
        """
        https://stackoverflow.com/a/2158532/4212158.
        """
        for element in irregular_list:
            if isinstance(element, (list, tuple, set)) and not isinstance(
                element, (str, bytes, dict)
            ):
                yield from flatten_list(element)  # recurse
            else:
                yield element

    return list(flatten_generator(list_))


@beartype
def str_to_bool(s: Union[str, bool]) -> bool:
    """
    Converts a string-bool to a Python bool.

    This is useful when we load Kedro bool params as a str
    """
    if isinstance(s, bool):
        return s
    elif isinstance(s, str):
        if s.lower() == "true":
            return True
        elif s.lower() == "false":
            return False
        else:
            raise ValueError(f"unexpected value {s}")


@beartype
def bool_to_str(b: Union[str, bool]) -> str:
    assert str(b).lower() in ["true", "false"]
    if isinstance(b, str):
        return b
    elif isinstance(b, bool):
        return str(b).lower()


def get_id(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


@beartype
def is_dict_equal(D1: Dict, D2: Dict) -> bool:
    return (
        len(D1) == len(D2)
        and all([x in D1 for x in D2])
        and all([D1[k] == D2[k] for k in D1])
    )


@beartype
def assert_list_elements(L1: List, L2: List) -> bool:
    return len(L1) == len(L2) and all([x in L1 for x in L2])


@beartype
def all_elements_same(items: List) -> bool:
    return all(x == items[0] for x in items)


@beartype
def generate_hash(input_value: Union[int, str], length: int = 10) -> str:
    """
    Generates a random sha using input value.
    """
    hash = hashlib.sha1()
    hash.update(str(input_value).encode("utf-8"))
    return hash.hexdigest()[:length]


@beartype
def format_date(date: str) -> datetime:
    date_format_string = "%Y-%m-%dT%H:%M:%SZ"
    date_format_string_no_z = "%Y-%m-%dT%H:%M:%S.%f"
    date_format_string_dot_z = "%Y-%m-%dT%H:%M:%S.%fZ"
    if "." in date and "Z" in date:
        formatted = datetime.strptime(date, date_format_string_dot_z)
    elif "Z" in date:
        formatted = datetime.strptime(date, date_format_string)
    else:
        formatted = datetime.strptime(date, date_format_string_no_z)
    return formatted


@beartype
def pprint_list(items: list[Any]) -> str:
    return pprint.pformat(natural_sort(items))


def pandas_print_everything():
    """
    Sets global pandas print options to print everything.
    """
    import pandas as pd

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)


@beartype
def kwargs_to_filename(
    date: DateType = None,
    ext="",
    suffix: Optional[str] = None,
    key_only: Optional[list[str]] = None,
    **kwargs,
) -> str:
    """
    Generates a nice filename from a dictionary of kwargs.

    Args:
        date_format: used to optionally add a date to the filename.
        key_only: if present, only the key is used, not the value.
        suffix: if present, will add a suffix to the filename.
        ext: if present, will add an extension to the filename.
        **kwargs: a dictionary of kwargs to turn into a file name

    Returns:
        a filename string encoding all the kwargs

    Examples:
        >>> kwargs_to_filename(a=1, b=2, date=None)
    """
    filename = []

    if date == "datetime":
        VERSION_FORMAT = "%Y-%m-%dT%H-%M-%S-%f"
        current_ts = datetime.now(tz=timezone.utc).strftime(VERSION_FORMAT)
        timestr = current_ts[:-5] + current_ts[-1:]  # Don't keep microseconds
        filename.append(timestr)
    elif date == "date":
        VERSION_FORMAT = "%Y-%m-%d"
        current_ts = datetime.now(tz=timezone.utc).strftime(VERSION_FORMAT)
        filename.append(current_ts)

    if key_only:
        for k in key_only:
            filename.append(k)

    kwargs = natural_sort(kwargs)  # deterministic order

    for k, v in kwargs.items():
        try:  # for the regex later
            v = v.strip()
        except AttributeError:
            pass
        filename.append(f"{k}={v}")

    assert filename, "filename is empty"
    filename = "+".join(filename)
    """
    Edge cases.
    """
    # replace any illegal characters with a space using regex and a blacklist:
    # / \ : * ? " < > |
    blacklisted = re.compile(r"[\s\\/:*?\"<>|]")
    filename = blacklisted.sub(" ", filename)

    # if any of the kwargs are empty, remove them
    # e.g. "a=1+ + + + +c=5" -> "a=1+c=5"
    filename = filename.replace("+ +", "")
    # the above doesn't work,

    # clean up double spaces
    filename = re.sub(r"\s+", " ", filename)

    # clean up ++
    filename = filename.replace("++", "+")

    # remove leading and trailing spaces or +
    filename = filename.strip()
    filename = filename.strip("+")

    if suffix:
        filename = f"{filename}__{suffix}"
    if ext:
        filename = f"{filename}.{ext}"

    return filename


def fname_to_kwargs(fname: str) -> Dict[str, Any]:
    """
    Converts a filename to a dictionary of kwargs.

    Args:
        fname: a filename string

    Returns:
        a dictionary of kwargs

    Examples:
        >>> fname_to_kwargs("a=1+b=2")
        {"a": 1, "b": 2}
    """
    # remove extension
    fname = fname.split(".")[0]
    # remove suffix
    fname = fname.split("__")[0]

    kwargs = {}
    for part in fname.split("+"):
        try:
            k, v = part.split("=")
            kwargs[k] = v
        except ValueError:
            continue
    return kwargs
