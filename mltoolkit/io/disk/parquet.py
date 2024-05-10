"""
stolen from https://jpweytjens.be/read-multiple-files-with-pandas-fast/
"""

from functools import partial

import pandas as pd
import pyarrow as pa
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map


def _read_parquet(filename, columns=None, filters=None):
    """
    Wrapper to pass to a ProcessPoolExecutor to read parquet files as fast as possible. The PyArrow engine (v4.0.0) is
    faster than the fastparquet engine (v0.7.0) as it can read columns in parallel. Explicitly enable multithreaded
    column reading with `use_threads == true`.

    Parameters
    ----------

    filename : str
        Path of the parquet file to read.
    columns : list, default=None
        List of columns to read from the parquet file. If None, reads all columns.

    Returns
    -------
    pandas Dataframe
    """

    return pd.read_parquet(
        filename, columns=columns, engine="pyarrow", use_threads=True, filters=filters
    )


def read_parquet(
    files,
    columns=None,
    parallel=True,
    n_concurrent_files=8,
    n_concurrent_columns=4,
    show_progress=True,
    ignore_index=False,
    chunksize=None,
    filters=None,
):
    """
    Read a single parquet file or a list of parquet files and return a pandas DataFrame.

    If `parallel==True`, it's on average 50% faster than `pd.read_parquet(..., engine="fastparquet")`.

    Limited benchmarks indicate that the default values for `n_concurrent_files` and `n_concurrent_columns` are the
    fastest combination on a 32 core CPU.
    `n_concurrent_files` * `n_concurrent_columns` <= the number of available cores.

    Parameters
    ----------

    files : list or str
        String with path or list of strings with paths of the parqiaet file(s) to be read.
    columns : list, default=None
        List of columns to read from the parquet file(s). If None, reads all columns.
    parallel : bool, default=True
        If True, reads both files and columns in parallel. If False, read the files serially while still reading the columns in parallel.
    n_concurrent_files : int, default=8
        Number of files to read in parallel.
    n_concurrent_columns : int, default=4
        Number of columns to read in parallel.
    show_progress : bool, default=True
        If True, shows a tqdm progress bar with the number of files that have already been read.
    ignore_index : bool, default=True
        If True, do not use the index values along the concatenation axis. The resulting axis will be labeled 0, ..., n-1. This is useful if you are concatenating objects where the concatention axis does not have meaningful indexing information.
    chunksize : int, default=None
        Number of files to pass as a single task to a single process. Values greater than 1 can improve performance if each task is expected to take a similar amount of time to complete and `len(files) > n_concurrent_files`. If None, chunksize is set to `len(files) / n_concurrent_files` if `len(files) > n_concurrent_files` else it's set to 1.

    Returns
    ------
    pandas DataFrame
    """

    # ensure files is a list when reading a single file
    if isinstance(files, str):
        files = [files]

    # no need for more cpu's then files
    if len(files) < n_concurrent_files:
        n_concurrent_files = len(files)

    # no need for more workers than columns
    if columns:
        if len(columns) < n_concurrent_columns:
            n_concurrent_columns = len(columns)

    # set number of threads used for reading the columns of each parquet files
    pa.set_cpu_count(n_concurrent_columns)

    # try to optimize the chunksize based on
    # https://stackoverflow.com/questions/53751050/python-multiprocessing-understanding-logic-behind-chunksize
    # this assumes each task takes roughly the same amount of time to complete
    # i.e. each dataset is roughly the same size if there are only a few files
    # to be read, i.e. ´len(files) < n_concurrent_files´, give each cpu a single file to read
    # when there are more files than cpu's give chunks of multiple files to each cpu
    # this is in an attempt to minimize the overhead of assigning files after every completed file read
    if (chunksize is None) and (len(files) > n_concurrent_files):
        chunksize, remainder = divmod(len(files), n_concurrent_files)
        if remainder:
            chunksize += 1
    else:
        chunksize = 1

    if parallel is True and n_concurrent_files > 1:
        if show_progress:
            print(
                f"Reading {len(files)} parq files in parallel using {n_concurrent_files} threads"
            )
        _read_parquet_map = partial(_read_parquet, columns=columns, filters=filters)
        dfs = process_map(
            _read_parquet_map,
            files,
            max_workers=n_concurrent_files,
            chunksize=chunksize,
            disable=not show_progress,
        )

    else:
        dfs = [
            _read_parquet(file, columns=columns, filters=filters)
            for file in tqdm(files, desc="parq shards", disable=not show_progress)
        ]

    # reduce the list of dataframes to a single dataframe
    df = pd.concat(dfs, ignore_index=ignore_index)

    return df
