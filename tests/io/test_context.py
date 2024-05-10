import time

import pytest

from decalmlutils.io.context import DeadlineExceededException, deadline


def test_deadline():
    @deadline(1e-2)  # 1e-3 flakey with Bitbucket CI bc of their slow machines
    def my_function():
        time.sleep(1e-5)
        return True

    # This should not raise an exception
    assert my_function() is True

    @deadline(1e-4)
    def my_function():
        time.sleep(1e-3)
        return True

    # This should raise a DeadlineExceededException
    with pytest.raises(DeadlineExceededException):
        my_function()
