import logging
import os
import shutil
import signal
import tempfile
import time
from contextlib import contextmanager

from beartype import beartype
from beartype.typing import Union

logger = logging.getLogger(
    __name__
)  # do not use cloudwatch logging here due to circular imports


class safe_open(object):
    """
    Context manager to ensure file writing is atomic.

    based on: https://youtu.be/aGPi27VGWVM?t=1214
    """

    def __init__(self, path, mode="w+b", extension=None):
        assert "/" in path, f'path must contain a "/" character: {path}'
        self._target = path
        self._mode = mode
        self._extension = extension

    def __enter__(self):
        self._file = tempfile.NamedTemporaryFile(
            mode=self._mode, suffix=self._extension, delete=False
        )
        return self._file

    def __exit__(self, exc_type, _exc_value, traceback):
        self._file.close()
        if exc_type:  # write interrupted
            os.unlink(self._file.name)
        else:
            dir, _fname = self._target.rsplit("/", 1)
            os.makedirs(dir, exist_ok=True)
            # this line failed when moving files across filesystems
            # os.rename(self._file.name, self._target)
            shutil.move(self._file.name, self._target)


def atomic_copy(src: str, dst: str):
    """
    Copy a file from src to dst, ensuring that the copy is atomic.

    Args:
        src: source file path
        dst: destination file path

    Returns:
        None
    """
    with safe_open(dst) as f:
        shutil.copyfileobj(open(src, "rb"), f)


class DisableKeyboardInterrupt:
    """
    Context manager to prevent keyboardinterrupt from interrupting important code.

    source: https://stackoverflow.com/a/21919644/4212158
    """

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)
        self.num_attempts = 0
        self.max_attempts = 10
        # set os env var that signals we haven't received a keyboard interrupt
        os.environ["KEYBOARD_INTERRUPTED"] = "0"

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)

        if self.num_attempts >= self.max_attempts:
            logger.info("Max attempts reached. Interrupting code block.")
            self.old_handler(*self.signal_received)
            os.environ[
                "KEYBOARD_INTERRUPTED"
            ] = "1"  # signals downstream code that we received a keyboard interrupt
        else:
            logger.info(
                "Keyboard interrupt has been disabled! If you want to kill this code block, look up the process's "
                f"process ID and run `kill -INT <pid>`. Or, just repeat the keyboard interrupt "
                f"{self.max_attempts-self.num_attempts} more times."
            )

        self.num_attempts += 1

    def __exit__(self, type, value, traceback):
        pass  # <-- ignore keyboard interrupt
        # below lines delay the keyboard interrupt rather than disabling it
        # signal.signal(signal.SIGINT, self.old_handler)
        # if self.signal_received:
        #     self.old_handler(*self.signal_received)


class DeadlineExceededException(Exception):
    pass


@contextmanager
@beartype
def deadline(timeout_seconds: Union[int, float]):
    """
    Contextmanager that gives a deadline to run some code.

    Usage:
        `with deadline(secs):`

        or

        ```
        @deadline(secs)
        def func: ...
        ```

    Args:
        timeout_seconds: number of seconds before context manager raises a DeadlineExceededException

    Raises:
        DeadlineExceededException error if more than timeout_seconds elapses.
    """
    start_time = time.time()

    # https://stackoverflow.com/questions/492519/timeout-on-a-function-call
    def timeout_handler(_signum, _frame):
        elapsed_time = time.time() - start_time
        msg = f"Deadline of {timeout_seconds} seconds exceeded by {elapsed_time - timeout_seconds:.1f} seconds"
        logger.exception(msg)
        raise DeadlineExceededException(msg)

    # set the timeout handler
    signal.signal(
        signal.SIGALRM, timeout_handler
    )  # allows handler to be executed when signal is received
    # signal.alarm(timeout_seconds) # sets the alarm to go off after timeout_seconds. must be int
    # to permit alarms with floats instead of ints, we could use:
    signal.setitimer(
        signal.ITIMER_REAL, timeout_seconds
    )  # ITIMER_REAL will raise SIGALRM
    try:
        yield
    finally:
        signal.alarm(0)  # cancel alarm if the context manager exits before the timeout
