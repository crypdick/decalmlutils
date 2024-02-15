import os


from rd_utils.conf import settings
from rd_utils.io.aws.s3_local import local_fpath_to_s3url, s3url_to_local_mirror


def test_s3_to_local_fpath_symmetric():
    """
    Tests that the s3_to_local_fpath function is symmetric to the local_to_s3_fpath function.
    """
    s3url = f"s3://{settings.ML_BUCKET}/path/to/my_key.png"

    local_fpath = s3url_to_local_mirror(s3url)

    expected_local_fpath = os.path.join(settings.LOCAL_DATA_DIR, "path/to/my_key.png")
    assert local_fpath == expected_local_fpath

    # now, test the reverse
    s3url_result = local_fpath_to_s3url(local_fpath)
    assert s3url_result == s3url
