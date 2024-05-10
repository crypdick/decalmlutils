"""
Tools for downloading individual S3 objects to local disk.

for downloading many files, avoid recreating S3 client for each individual transfer; use s3_client_ray.S3ClientActor()
instead (which wraps these)
"""

import io
import json
import logging
import os
import pathlib
import shutil
from math import ceil

import pandas as pd
import ray
from beartype import beartype
from beartype.typing import Any, Dict, List, NamedTuple, Optional, Tuple
from metaflow import S3
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm import tqdm

from decalmlutils.conf import settings
from decalmlutils.io.aws.s3 import __get_file_bytes_from_s3, split_s3url
from decalmlutils.io.disk.img import (
    convert_image_type,
    create_dummy_img,
    img_bytes_to_Image,
    write_img_to_disk,
)
from decalmlutils.io.disk.misc import (
    create_dir_if_not_exists,
    delete_file_if_exists,
    file_exists,
    read_bytes_from_disk,
    write_bytes_to_disk,
)
from decalmlutils.io.disk.parquet import read_parquet
from decalmlutils.tensors import get_chunks

# do not use get_cloudwatch_logger here-- it will cause Ray pickling errors
logger = logging.getLogger(__name__)


class Payload(NamedTuple):
    """
    A typed dataclass for all requests.
    """

    s3url: str
    data: Any
    destination_fpath: Optional[str] = None
    metadata: Optional[dict] = None


@beartype
def check_get_metaflow(
    s3url: str,
    check_cache: bool = True,
    output_fpath: Optional[str] = None,
    client_params: Optional[Dict] = None,
) -> str:
    """
    Version of check_get which uses Metaflow's S3 client, which they claim is fast.

    In practice, the initialization step is pretty slow. However, it may be faster than boto3 for very large files, such
    as model weights. I haven't benchmarked this.

    Args:
        s3url:          URL to pull from s3
        check_cache:    Check for cached object before pulling fresh file
        output_fpath:   Destination for the downloaded object
        client_params:  Additional parameters passed onto the boto3 client setup

    Returns:
        output_fpath:   Local location of file
    """
    s3url = s3url.strip()  # fix trailing space problems
    if s3url.endswith("/"):
        raise ValueError(f"Passed in a directory instead of a object url: {s3url}")

    if not output_fpath:
        output_fpath = s3url_to_local_mirror(s3url)
        create_dir_if_not_exists(output_fpath)

    to_download = True

    if check_cache:
        # avoid slow I/O if not using cache, as with speed tests
        destination_fpath = check_s3url_already_cached(s3url)
        if destination_fpath:  # already cached
            to_download = False

    if to_download:
        with S3(client_params=client_params) as s3:
            s3obj = s3.get(s3url)
            shutil.move(s3obj.path, output_fpath)

    return output_fpath


@beartype
def check_get(
    s3url: str,
    client: Optional[object] = None,
    check_cache: bool = False,
    store_copy: bool = False,
) -> Payload:
    """
    Generic method to get objects from S3 as bytes, which can optionally check if pre-cached before downloading.

    This method is wrapped by other functions when post-processing bytes into an object is appropriate.

    Args:
        s3url: url of the object to get
        client: (optional) an instantiated boto3 client. If not given, a new one will be spawned.
        check_cache: whether to check the local cache for the object. This can cause disk thrashing when reading many
            small files, so disable if you are sure ahead of time that the objects are not pre-cached. Disable when
            running speed tests.
        store_copy: whether to store a copy of the object. Disable when using this on Lambda.

    Returns:
        The requested object packaged in a Payload dataclass.
    """
    s3url = s3url.strip()  # fix trailing space problems
    if s3url.endswith("/"):
        raise ValueError(f"Passed in a directory instead of a object url: {s3url}")
    bytes_arr = None
    destination_fpath = s3url_to_local_mirror(s3url)
    precached = False

    # Avoid slow I/O if not using cache, as with speed tests.
    # It is faster to read from disk just once instead of looking before we leap
    if check_cache:
        try:
            bytes_arr = read_bytes_from_disk(destination_fpath)
            precached = True
        except FileNotFoundError:
            pass

    if bytes_arr is None:
        # either not in cache or cache=False; fetch from S3
        logger.debug(f"getting {s3url} from s3")
        bytes_arr = __get_file_bytes_from_s3(s3url, client=client)

    if store_copy and bytes_arr and not precached:
        # copy bytes to disk before return
        logger.debug(f"saving {s3url} to {destination_fpath}")
        write_bytes_to_disk(bytes_arr, destination_fpath)
    else:
        logger.debug("keeping downloaded copy in memory (no cache)")

    # returns s3url in case we use as mapper
    return Payload(s3url=s3url, data=bytes_arr, destination_fpath=destination_fpath)


@retry(
    wait=wait_fixed(0.01),
    reraise=True,
    retry=retry_if_exception_type((OSError, TypeError)),
    stop=stop_after_attempt(3),
)
@beartype
def check_get_img(
    s3url: str,
    check_cache=True,
    store_copy=False,
    client: Optional[object] = None,
    destination_fpath=None,
    return_mock_data: bool = False,
    mode: Optional[str] = "RGB",
    as_type: str = "image",
) -> Payload:
    """
    Wrapper for check_get which post-processes imgs bytes into PIL.Image.

    If it tries to open a pre-cached image file that is corrupted, it will automatically delete it and re-download it.
    """
    if return_mock_data:
        return Payload(
            s3url=s3url,
            data=create_dummy_img(mode, as_type),
            destination_fpath="/dummy/path.png",
        )

    # NOTE: never use check_get(store_copy=True)-- need to rm alpha channel first!
    # we will store a copy in this function if store_copy=True
    payload = check_get(s3url, check_cache=check_cache, store_copy=False, client=client)
    img_bytes = payload.data

    if not img_bytes:  # s3 object did not exist, do not retry
        img, destination_fpath = None, None
    else:
        try:
            if destination_fpath is None:
                destination_fpath = s3url_to_local_mirror(
                    s3url
                )  # NOTE: keep here so that os.remove() works
            # NOTE: do not use read_img_from_disk here
            img = img_bytes_to_Image(img_bytes, mode=mode)

            if store_copy:
                write_img_to_disk(img, destination_fpath)
        # handle different failure modes
        except IndexError as e:
            logger.warning(f"Image should have 3 channels: {s3url}")
            raise e
        except (OSError, TypeError) as e:
            if check_cache:
                logger.warning(
                    f"""
           Failed to open {s3url} from cache dir {destination_fpath}, error {e}
           Deleting cached copy (if one exists) and retrying s3 fetch.
           """
                )
            delete_file_if_exists(destination_fpath)
            raise e  # handled by @retry
        except Exception as e:
            logger.warning(f"Failed to load {s3url} with exception {e}")
            raise e

    img = convert_image_type(img, as_type=as_type)

    return Payload(s3url=s3url, data=img, destination_fpath=destination_fpath)


@beartype
def check_get_json(s3url, check_cache=True, store_copy=False, client=None) -> Payload:
    """
    Wrapper for check_get which post-processes JSON bytes into a Python dict.
    """
    payload = check_get(
        s3url, check_cache=check_cache, store_copy=store_copy, client=client
    )
    data = json.load(io.BytesIO(payload.data))
    # https://stackoverflow.com/a/53709437/4212158
    payload = payload._replace(
        data=data
    )  # AttributeError: can't set attribute if try to overwrite payload.data directly
    return payload


@beartype
def check_s3url_already_cached(s3url: str) -> Optional[str]:
    """
    Check if an s3 object is already stored on this machine locally.
    """
    cache_fpath = s3url_to_local_mirror(s3url)

    if file_exists(cache_fpath):
        return cache_fpath
    else:
        return None


@beartype
def s3url_to_local_mirror(s3url: str, data_root=settings.LOCAL_DATA_DIR) -> str:
    """
    converts an s3url to the local dir it belongs to. This assumes that the object urls use a similar data structure
    that we want to see locally, i.e. s3://ml-bucket/01_raw/file.ext will go to data/01_raw/file.ext.

    if the bucket is the tiles bucket, always save the objects in IMG_DIR instead

    See also: local_fpath_to_s3url
    """
    _bucket, prefix_and_key = split_s3url(s3url)

    assert (
        prefix_and_key is not None
    ), f"Expected prefix_and_key to not be Nonetype. S3 url must be malformed: {s3url}"

    # mirror dir structure of bucket inside data/
    mirrored_cache_path = os.path.join(data_root, prefix_and_key)
    return mirrored_cache_path


@beartype
def local_fpath_to_s3url(local_fpath: str, data_root=settings.LOCAL_DATA_DIR) -> str:
    """
    Given a local data fpath, return the corresponding s3 url (in the Kedro ML bucket)

    See also: s3url_to_local_mirror
    """
    assert local_fpath.startswith(
        data_root
    ), f"local_fpath must be a path inside {data_root}"

    # local_fpath_relative = local_fpath.removeprefix(LOCAL_DATA_DIR)  # todo when anyscale updates their python
    local_fpath_relative = local_fpath[len(data_root) :]
    if local_fpath_relative.startswith("/"):
        local_fpath_relative = local_fpath_relative[1:]
    # add the ML bucket prefix
    s3url = f"s3://{settings.ML_BUCKET}/{local_fpath_relative}"

    return s3url


@beartype
def filter_cached_files(
    s3urls: List[str], hide_progress: bool = False, data_root=settings.LOCAL_DATA_DIR
) -> List[str]:
    """
    Given list of S3 urls, returns list of files that are not already downloaded.
    """
    if not hide_progress:
        logger.info(f"Scanning {data_root} for cached files")
    files_to_download = []
    already_downloaded = 0

    def use_full_scan() -> bool:
        if len(s3urls) > 1_000:
            return True
        elif len(s3urls) < 100:
            return False

        # check if data_root has > 1k files, and to abort the scan early if it does
        count = 0
        for _ in os.scandir(data_root):
            count += 1
            if count > 1_000:
                return True
        return False

    if use_full_scan():  # method 1 is faster for large lists
        cached_files = {
            str(abspath) for abspath in pathlib.Path(data_root).glob("**/*")
        }
        for url in tqdm(
            s3urls,
            unit=" files checked",
            desc="Checking if files exist",
            disable=hide_progress,
        ):
            local_img = s3url_to_local_mirror(url)
            if local_img not in cached_files:
                files_to_download.append(url)
            else:
                already_downloaded += 1
    else:  # method 2 is faster if there are few s3urls to check
        for url in s3urls:
            local_fpath = s3url_to_local_mirror(url)
            if not os.path.isfile(local_fpath):
                files_to_download.append(url)
            else:
                already_downloaded += 1

    if not hide_progress:
        logger.info(
            f"Out of a total {len(s3urls)} files, found {already_downloaded} files already downloaded. "
            f"{len(files_to_download)} remaining files to download"
        )
    return files_to_download


@beartype
def get_many_s3_files(
    s3urls: List[str],
    check_cache: bool = True,
    missing_ok: bool = True,
    chunksize=10_000,
    hide_progress: bool = False,
    mock_download: bool = False,
    client_params: Optional[Dict] = None,
) -> Tuple[List[str], List[str]]:
    """
    Downloads many files from S3 in parallel, without checking if they already exist.

    All files are saved to the local mirror.

    Args:
        s3urls:         List of S3 urls to download.
        check_cache:    Whether to check if files already exist in local mirror
        missing_ok:     Whether to raise an error if a file is missing
        chunksize:      The max number of files to download in one request
        hide_progress:  Hide the progress bar
        mock_download:  Mock downloading files
        client_params:  S3 params to pass to the boto3 client


    Returns:
        Tuple of (downloaded_files, missing_files), where each is a list of S3 urls.
    """
    assert len(s3urls) > 0

    if mock_download:
        logger.info("Mock downloading files")
        return s3urls, []

    if check_cache:
        files_to_download = filter_cached_files(s3urls, hide_progress=hide_progress)
        if len(files_to_download) == 0:
            return s3urls, []
    else:
        files_to_download = s3urls

    # if file does not start with s3://, add the prefix
    files_to_download = [
        s3url if s3url.startswith("s3://") else f"s3://{s3url}"
        for s3url in files_to_download
    ]

    chunks = get_chunks(len(files_to_download), chunk_size=chunksize)

    if not hide_progress:
        logger.info(
            f"downloading files over {len(chunks)} chunks. this might take a long time!"
        )
    success, failed = [], []
    with S3(client_params=client_params) as s3c:
        # smoothing=0 to disable exponential smoothing
        with tqdm(
            total=len(files_to_download),
            unit=" files",
            desc="Downloading files",
            disable=hide_progress,
            smoothing=0,
        ) as pbar:
            for chunk in chunks:
                chunk_files = files_to_download[chunk[0] : chunk[1]]
                # return missing to not throw exception if the file does not exist on S3
                s3objs = s3c.get_many(chunk_files, return_missing=missing_ok)

                # mv tmp files to the local mirror
                for s3obj in s3objs:
                    destination_fpath = s3url_to_local_mirror(s3obj.url)

                    create_dir_if_not_exists(destination_fpath)

                    try:
                        # move the tmp file to the local mirror. if the destination is on the same filesystem, this uses os.rename, which is fast, atomic, and uses less space
                        shutil.move(s3obj.path, destination_fpath)
                        success.append(s3obj.url)
                    except TypeError:  # img missing
                        # if we use the missing_ok option, the s3obj.path will be None for non-existent S3 objects
                        failed.append(s3obj.url)
                        continue
                    except OSError as e:
                        logger.error(
                            f"Error copying file {s3obj.path} to {destination_fpath}: {e}"
                        )
                        raise e
                    finally:
                        pbar.update()

    # if this script is interrupted, it'll orphan a `metaflow.s3.*` directory. this is a hack to clean it up
    for p in pathlib.Path("../io/aws").glob("metaflow.s3.*"):
        shutil.rmtree(p)

    return success, failed


def load_s3_parquet(
    shard_s3urls: List[str],
    keep_cache: bool = True,
    use_metaflow_s3: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Given a list of parquet shard urls, load them and concatenate them into a single dataframe.

    Args:
        shard_s3urls: list of prediction shard urls
        keep_cache: whether to keep the cached files after loading
        **kwargs: Parquet kwargs

    Returns:
        the parquet shards loaded into a single pd.Dataframe
    """
    # keep a local copy in case we need to re-download
    # Metaflow's S3 client is also more fault-tolerant the wrangler's
    # however, this is not good for instances with small disks because it will throw an OSError. in that case, do not use metaflow's S3 client
    if use_metaflow_s3 and keep_cache:
        if len(shard_s3urls) == 1:
            logger.info(
                "Using Metaflow's S3 client to load single shard. Consider using without metaflow S3 client."
            )
        get_many_s3_files(
            shard_s3urls, missing_ok=False, hide_progress=True, check_cache=True
        )
        # read cached files and keep the index
        df = read_parquet(
            [s3url_to_local_mirror(url) for url in shard_s3urls],
            ignore_index=False,
            show_progress=False,
            **kwargs,
        )
        if not keep_cache:
            [delete_file_if_exists(s3url_to_local_mirror(url)) for url in shard_s3urls]
    else:
        if keep_cache:
            logger.warning(
                "keep_cache=True is not supported when use_metaflow_s3=False or single shards"
            )
        df = read_parquet(
            shard_s3urls, ignore_index=False, show_progress=False, **kwargs
        )

    return df


@ray.remote
def _wrapped_get_many(*args, **kwargs):
    """
    Wraps get_many_s3_files, so that we can use it in a Ray pool.
    """
    return get_many_s3_files(*args, **kwargs)


def parallel_get_many_s3(
    s3urls: List[str], check_cache: bool = True, missing_ok: bool = False
):
    """
    Thin wrapper for get_many_s3_files that allows for parallelization.

    Usage:
        >>> ray.init()
        >>> parallel_get_many_s3(s3urls)
    """
    num_nodes = len(ray.nodes())
    chunksize = ceil(len(s3urls) / num_nodes)
    chunks = get_chunks(len(s3urls), chunk_size=chunksize)
    url_chunks = [s3urls[chunk[0] : chunk[1]] for chunk in chunks]
    # get number of nodes in the cluster from Ray
    ray.get(
        [
            _wrapped_get_many.remote(
                s3url_chunk,
                check_cache=check_cache,
                missing_ok=missing_ok,
                chunksize=chunksize,
            )
            for s3url_chunk in url_chunks
        ]
    )
