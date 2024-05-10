import csv
import io
import logging
import os
import pickle
from contextlib import suppress

from beartype import beartype
from beartype.typing import Any

from decalmlutils.io.context import safe_open

# note: do not use get_cloudwatch_logger() here. it will cause Ray serialization errors
logger = logging.getLogger(__name__)


@beartype
def read_bytes_from_disk(fpath: str) -> bytes:
    with open(fpath, "rb") as cached:
        fh = io.BytesIO(cached.read())
        fh.seek(0)
        bytes_ = fh.read()
    logger.debug(f"loaded bytes object {fpath}")
    return bytes_


@beartype
def read_csv_from_disk(fpath: str) -> list:
    with open(fpath, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        data = [column for line in reader for column in line]
        return data


@beartype
def write_bytes_to_disk(bytes_: bytes, outpath: str):
    create_dir_if_not_exists(outpath)
    with safe_open(outpath, "wb") as out:
        out.write(bytes_)
    logger.debug(f"wrote bytes to {outpath}")


@beartype
def write_pickle(pkl: Any, outfile: str) -> None:
    if not outfile.endswith(".pkl"):
        outfile = outfile + ".pkl"

    create_dir_if_not_exists(outfile)

    with safe_open(outfile, "wb") as f:
        pickle.dump(pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"pickled! ||{outfile}")


@beartype
def read_pickle(infile: str):
    with open(infile, "rb") as f:
        unpickled = pickle.load(f)
    logger.info(f"loaded pickled object {infile}")
    return unpickled


@beartype
def file_exists(fpath: str) -> bool:
    return os.path.isfile(fpath)


@beartype
def ensure_dir(name: str):
    """
    Given a path to a directory, creates it if it doesn't exist.
    """
    os.makedirs(name, exist_ok=True)


@beartype
def create_dir_if_not_exists(fpath: str) -> None:
    """
    Given a path to a file name, creates the parent directory if it does not exist.
    """
    try:
        pardir, _ = fpath.rsplit("/", 1)  # rm filename
    except ValueError:
        pardir = "."
    os.makedirs(pardir, exist_ok=True)


@beartype
def delete_file_if_exists(fpath: str) -> None:
    with suppress(FileNotFoundError):
        os.remove(fpath)
