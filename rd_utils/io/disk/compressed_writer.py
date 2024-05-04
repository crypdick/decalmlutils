import gzip
import os
from filelock import FileLock


class GzipWriter:
    """
    Incrementally write text to a gzip file.

    Useful for writing large files that would not fit on disk uncompressed.

    Args:
        fpath: the path to the gzip file to write to
        buffer_limit: the number of lines to buffer before writing to disk. Larger values
            will use more memory but will achieve better compression, since gzip compresses
            each chunk independently.
        check_exists: if True, will raise an error if the file already exists
        mode: the mode to open the file in. Default is "at" (append text).
    """

    def __init__(self, fpath, buffer_limit=10_000, check_exists=True, mode="at"):
        self.fpath = fpath
        self.buffer = []
        self.buffer_limit = buffer_limit
        self.mode = mode

        if check_exists:
            assert not os.path.exists(fpath), f"File {fpath} already exists"

    def append(self, text: str, append_newline=True, flush=False) -> None:
        if append_newline and not text.endswith("\n"):
            text += "\n"
        self.buffer.append(text)
        if flush:
            self.flush()
        else:
            self.check_flush()

    def check_flush(self):
        if len(self.buffer) >= self.buffer_limit:
            self.flush()

    def flush(self):
        with FileLock(str(self.fpath) + ".lock"):  # make thread-safe
            with gzip.open(self.fpath, self.mode) as f:
                f.write("".join(self.buffer))
            self.buffer = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()
        return False
