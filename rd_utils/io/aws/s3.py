import io
import logging
import os
import time

import metaflow
from beartype import beartype
from beartype.typing import Any, Collection, Dict, Optional, Tuple
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError, ParamValidationError
from metaflow import S3
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_fixed,
    wait_random_exponential,
)

from rd_utils.io.aws import get_aws_client


logger = logging.getLogger(__name__)


def upload_directory(
    upload_dir,
    s3root=None,
    tmproot=".",
    bucket=None,
    prefix=None,
    run=None,
    filter_func=None,
):
    """
    This function has been copied from dendra-pipeline-ng/src/flow_utils.py
    https://bitbucket.org/biocarbonengineering/dendra-pipeline-
    ng/src/9b04598d6c6825064df61f4de6ea4d6730f2875e/src/flow_utils.py#lines-170.

    :param upload_dir: path to local dir to upload
    :param s3root: the s3 URL (s3://bucket/key...) to use as the "root"
    :param filter_func: if given is called with each path before upload. If True
            is returned the file is uploaded, otherwise the file is ignored.
    :return: return value of metaflow.s3.put_files()
    """
    with S3(
        s3root=s3root, tmproot=tmproot, bucket=bucket, prefix=prefix, run=run
    ) as s3:

        def key_path_gen():
            for root, _, filenames in os.walk(upload_dir):
                for fname in filenames:
                    path = os.path.join(root, fname)

                    if filter_func is None or filter_func(path):
                        # create a key that is relative to outputs_dir, lstrip("/") removes the leading /
                        key = path.replace(upload_dir, "").lstrip("/")

                        yield (key, path)

        return s3.put_files(key_paths=key_path_gen())


def upload_to_s3(local_fpath: str, s3url: str, s3_client: Any = None) -> None:
    """
    Upload a file to S3.

    Args:
        local_fpath: local filepath to upload
        s3url: s3 url to upload to
        s3_client: boto3 s3 client to use. If None, a new client will be created.
    """
    s3_client = s3_client or get_aws_client("s3")
    bucket, object_name = split_s3url(s3url)

    try:
        s3_client.upload_file(local_fpath, bucket, object_name)
    except ClientError as e:
        logger.error(f"Error uploading {local_fpath} to {s3url}: {e}")
        raise e


@beartype
def split_s3url(s3url: str) -> Tuple[str, Optional[str]]:
    """
    Given a S3 url, split into bucket name and object key (everything after the bucket name).

    Will also work on urls missing a leading s3:// prefix

    Usage examples:
    split_s3url('s3://bucketname/path/to/file.ext') --> (bucketname, path/to/file.ext)
    split_s3url('s3://bucketname/path/to/dir/') --> (bucketname, path/to/dir/)
    split_s3url('bucketname/path/to/file.ext') --> (bucketname, path/to/file.ext)
    split_s3url('s3://bucketname') --> (bucketname, None)
    split_s3url('bucketname') --> (bucketname, None)
    """
    if s3url.startswith("s3://"):
        # throw out the s3:// .... prefix
        _protocol_prefix, bucket_key = s3url.split("s3://", 1)
    else:
        bucket_key = s3url

    try:
        s3_bucket, s3_key = bucket_key.split("/", 1)
    except ValueError:  # not enough vals to unpack
        # s3_bucket should be set to bucket_key, not s3url, which is the initial unstripped value.
        # if s3_bucket is set to s3url when initial value of s3url is 's3://bucket-name'
        # a ParamValidationError will be thrown when s3_bucket is passed to ls_s3
        s3_bucket = bucket_key
        s3_key = None

    assert s3_bucket is not None or s3_key is not None

    # NOTE: if original url is a dir rather than a specific object,
    # s3_key is a prefix that can be used by list_objects_v2
    return s3_bucket, s3_key


@beartype
def object_on_s3(
    object_url: str, s3_client: Optional[Any] = None, mock: bool = False
) -> bool:
    """
    Check whether a specific object url exists on S3.
    """
    bucket, key = split_s3url(object_url)

    if mock:
        return True

    if s3_client is None:
        s3_client = get_aws_client("s3")

    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


@beartype
def object_on_s3_many(
    object_urls: Collection[str], mock: bool = False
) -> Dict[str, bool]:
    """
    Like object_on_s3, but for multiple urls at once.
    """
    assert all(
        [s3url.startswith("s3://") for s3url in object_urls]
    ), "Must pass s3:// prefixed s3urls"

    if mock:
        return {url: True for url in object_urls}

    logger.info(f"Checking {len(object_urls)} objects on S3")
    start = time.time()
    with metaflow.S3() as s3:
        s3objs = s3.info_many(object_urls, return_missing=True)

        results = {s3obj.url: s3obj.exists for s3obj in s3objs}
        elapsed = time.time() - start
        logger.info(
            f"Checked {len(object_urls)} S3 urls in {elapsed:.1f} sec ({len(object_urls) / elapsed:.1f} urls/s)"
        )
        return results


@retry(
    # Wait at least 0.01 seconds + random up to 2^n * 0.1 seconds between each
    # retry until the range reaches 0.2 sec, then randomly up to 0.2 sec afterwards
    wait=wait_fixed(0.01) + wait_random_exponential(multiplier=0.1, max=0.2),
    reraise=True,
    retry=retry_if_exception_type(ClientError),
    stop=(stop_after_delay(20) | stop_after_attempt(10)),
)
@beartype
def __get_file_bytes_from_s3(
    s3url: str, client: Optional[object] = None
) -> Optional[bytes]:
    """
    The nucleus of all S3 client fetches.

    Has exponential back-off retry logic.

    Args:
        s3url: the object to fetch
        client: (optional) the client to use. If none given, a disposable one will be created.

    Returns:
        S3 object as bytes, or Nonetype if object does not exist.
    """
    bucket, key = split_s3url(s3url)

    if not client:
        client = get_aws_client("s3")

    try:
        with io.BytesIO() as fh:
            config = TransferConfig(max_concurrency=20)
            client.download_fileobj(bucket, key, fh, Config=config)
            fh.seek(0)  # NOTE: important to reset file cursor!
            bytes_arr = fh.read()

    except client.exceptions.NoSuchKey as e:
        logger.debug(f"No such key: {e.response}")
        bytes_arr = None  # do not retry

    except ParamValidationError as err:
        raise ValueError(f"The parameters you provided are incorrect: {err}")
    except ClientError as err:
        error_code = err.response.get("Error", {}).get("Code")
        if error_code == "404":
            logger.debug(
                f"S3 reports file not found (404): {s3url}. Ensure url is spelled correctly."
            )
            bytes_arr = None  # do not retry if 404
        else:
            raise err  # other ClientErrors errors (such as timeout) handled by retry

    return bytes_arr
