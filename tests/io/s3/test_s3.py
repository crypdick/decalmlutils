import pytest


@pytest.fixture
def bucket_name() -> str:
    """
    Return a dummy bucket name for testing purposes.
    """
    return "my-bucket"


@pytest.mark.skip
@pytest.mark.parametrize(
    "s3url",
    [
        "s3://my-bucket/my_key.png",
        "s3://my-bucket",
        "my-bucket",
        "s3://my-bucket/my_key.png/",
        "s3://my-bucket/",
        "my-bucket/",
    ],
)
def test_split_s3url(bucket_name, s3url):
    """
    Test decalmlutils.io.aws.misc.split_s3url.

    This is used in conjunction with ls_s3 for instance, in ls_s3_img. Passing empty strings and Nones to
    list_objects_v2 for the prefix is fine.
    """
    import decalmlutils.io.aws.s3 as io_s3

    bucket, key = io_s3.split_s3url(s3url)

    assert bucket == bucket_name
    assertion_dict = {
        "s3://my-bucket/my_key.png": key == "my_key.png",
        "s3://my-bucket": key is None,
        "my-bucket": key is None,
        "s3://my-bucket/my_key.png/": key == "my_key.png/",
        "s3://my-bucket/": key == "",
        "my-bucket/": key == "",
    }
    assert assertion_dict[s3url]


@pytest.mark.skip
def test_upload_directory(mocker):
    """
    Very basic sanity check.
    """
    import decalmlutils.io.aws.s3 as io_s3

    metaflowS3_mocker = mocker.patch.object(io_s3, "S3")

    io_s3.upload_directory("dummy_upload_dir", "dummy_s3_root")

    metaflowS3_mocker().__enter__().put_files.assert_called_once()
