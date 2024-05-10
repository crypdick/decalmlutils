import logging

import ray
from beartype import beartype
from beartype.typing import Any, Callable, Optional
from metaflow import S3
from tenacity import Retrying, wait_exponential
from tqdm import tqdm

from decalmlutils.conf import settings
from decalmlutils.io.aws import get_aws_client
from decalmlutils.io.aws.s3 import split_s3url

# do not use get_cloudwatch_logger here-- it will cause Ray pickling errors
logger = logging.getLogger(__name__)


@beartype
def is_png_file(key: str) -> bool:
    """
    Filter func for ls_s3_img for testing purposes,we want a named function for the assertion dict.
    """
    return key.lower().endswith(".png")


@beartype
def no_filter(_key: str) -> bool:
    """
    Filter func for ls_s3 for testing purposes, we want a named function for the assertion dict.
    """
    return True


@beartype
def ls_s3_multiprefix(
    prefixes, bucket: Optional[str] = None, filter_func: Callable = no_filter
) -> list[str]:
    """
    List all S3 objects in a list of prefixes.
    """
    logger.info(f"Recursively searching through {len(prefixes)} prefixes")
    s3urls = []
    with S3(bucket=bucket) as s3:
        # note: this for loop is blocked until metaflow finishes searching all the prefixes
        # it will appear as if nothing is happening and then all the results will appear at once
        for obj in s3.list_recursive(prefixes):
            s3url = obj.url
            if filter_func(s3url):
                s3urls.append(s3url)
                if len(s3urls) % 10_000 == 0:
                    logger.info(f"Progress: {len(s3urls)} objects found")

    logger.info(f"Found {len(s3urls)} objects in {len(prefixes)} prefixes")

    return s3urls


@beartype
def ls_s3_multiprefix_concurrent(
    prefixes, bucket: Optional[str] = None, filter_func: Callable = no_filter
) -> list[str]:
    """
    Version of ls_s3_multiprefix that uses concurrency and ls_s3.
    """
    logger.info(f"Recursively searching through {len(prefixes)} prefixes")
    s3urls = []

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for prefix in prefixes:
            futures.append(
                executor.submit(
                    ls_s3, prefix_dir=prefix, bucket=bucket, filter_func=filter_func
                )
            )
        for future in futures:
            s3urls.extend(future.result())
            logger.info(f"Progress: {len(s3urls)} objects found")

    logger.info(f"Found {len(s3urls)} objects in {len(prefixes)} prefixes")
    return s3urls


@beartype
def ls_s3(
    prefix_dir: Optional[str] = None,
    bucket: str = settings.ML_BUCKET,
    max_keys: int = -1,
    filter_func: Callable = no_filter,
    min_object_size: int = 0,
    client: Optional[Any] = None,
    region_name: Optional[str] = None,
    verbose: bool = True,
) -> list[str]:
    """
    Lists objects in an S3 bucket and filtering which ones we want.

    Args:
        client:             S3 client. If none is passed in, one will be instantiated.
        min_object_size:    Minimum object size (in bytes) to keep
        max_keys:           For terminating the list_objects operation early
        filter_func:        For filtering which objects to keep based on their s3 prefix and key
        bucket:             S3 bucket name
        prefix_dir:         In S3, a "prefix" is everything after s3://{bucket_name}/ and before {filenames.ext}. If None,
                            we will search the entire bucket.
        region_name:        Region Name to be provided to the aws client. Only applicable if client is None

    Returns:
        A list of S3 object urls
    """
    if client is None:
        client = get_aws_client("s3", region_name=region_name)

    list_objects_kwargs = {}

    if prefix_dir:
        if prefix_dir.startswith(
            "s3://"
        ):  # make this function robust to malformed inputs
            bucket2, prefix_dir = split_s3url(prefix_dir)
            if bucket2 != bucket:
                logger.warning(
                    f"Bucket set to {bucket} but prefix starts with {bucket2}. Overriding."
                )
                bucket = bucket2

        list_objects_kwargs["Prefix"] = prefix_dir

    # keep this down here so we can replace the bucket
    list_objects_kwargs["Bucket"] = bucket

    if verbose:
        logger.info(
            f"Paging thru  S3 bucket:{bucket}, prefix:{prefix_dir} and filtering for files of interest\n"
        )
    valid_keys = set()
    done = False
    with tqdm(unit=" pages", mininterval=5.0) as pbar_all, tqdm(
        unit=" valid objects found",
        mininterval=5.0,
        disable=max_keys < 10 or not verbose,
    ) as pbar_good:
        while not done:
            for attempt in Retrying(
                wait=wait_exponential(multiplier=1, min=0.001, max=10)
            ):
                with attempt:
                    resp = client.list_objects_v2(**list_objects_kwargs)
            if resp["KeyCount"] == 0:
                logger.info("Warning: 0 keys found.")
                done = True
                break
            pbar_all.update()
            for obj in resp["Contents"]:
                if obj["Size"] < min_object_size:
                    continue

                key = obj["Key"]
                if filter_func(key):
                    key = f"s3://{bucket}/{key}"
                    if key.endswith("/"):
                        continue  # sometimes S3 returns directories as objects
                    valid_keys.add(key)
                    pbar_good.update()

                if 0 < max_keys <= len(valid_keys):
                    done = True
                    break
            try:
                list_objects_kwargs["ContinuationToken"] = resp["NextContinuationToken"]
            except KeyError:
                done = True
                break

    if verbose:
        logger.info(f"Fetched {len(valid_keys)} keys")

    return list(valid_keys)


@beartype
def ls_s3_img(
    data_dir: str,
    max_keys: int = -1,
    client: Optional[Any] = None,
    mock_return: bool = False,
) -> list[str]:
    """
    Wraps ls_s3 and filters for files ending with .png extension.
    """
    if mock_return:
        return [f"{data_dir}/{i}" for i in range(max_keys)]

    # NOTE: data_dir expect not to end with an object
    bucket, prefix_dir = split_s3url(data_dir)
    img_urls = ls_s3(
        max_keys=max_keys,
        filter_func=is_png_file,
        bucket=bucket,
        prefix_dir=prefix_dir,
        client=client,
    )
    return img_urls


# use a small num CPU in case we are running smoke tests on a tiny node, so that we don't get resource errors
# otherwise, by default Ray will attempt to reserve 1 CPU for each task
@ray.remote(num_cpus=0.05)
@beartype
def ls_s3_img_ray(*args, **kwargs) -> list[str]:
    """
    Thin wrapper around ls_s3_img which turns into a Ray task.
    """
    # fixme: Mus this is broken again. however, using S3ClientActor.ls_s3_img() works
    # NOTE: need to init the client here or else we will get a TypeError: cannot pickle 'PyCapsule' object
    img_urls = ls_s3_img(*args, client=get_aws_client("s3"), **kwargs)
    return img_urls
