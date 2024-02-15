import logging
import logging.handlers
import sys
import traceback

from rd_utils.io.disk.misc import create_dir_if_not_exists

from .conf import settings

LEVEL = logging.INFO  # set global logging level


def setup_default_logging(level=LEVEL, log_path=settings.LOG_FPATH):
    logging.root.setLevel(level)

    console_handler = logging.StreamHandler()
    # metaflow alrdy logs asctime

    default_formatting = "%(name)s: [%(levelname)s] - %(message)s"
    default_formatter = logging.Formatter(default_formatting)
    console_handler.setFormatter(default_formatter)
    logging.root.addHandler(console_handler)

    if log_path:
        create_dir_if_not_exists(log_path)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=(1024**2 * 100), backupCount=4
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)

    # silence_boto_logging()

    def excepthook(exc_type, exc_message, stacktrace):
        """
        A logger exception hook for logging tracebacks and exceptions.

        Source: https://stackoverflow.com/a/9929970/4212158
        """
        logging.error(
            f"Logging an uncaught exception: {exc_message}. Traceback: \n{traceback.format_tb(stacktrace)}",
            exc_info=(exc_type, exc_message, stacktrace),
        )

    sys.excepthook = excepthook  # loggers will log exceptions before they are raised


def silence_boto_logging():
    """
    Silence boto logging.
    """
    logging.getLogger("boto3").setLevel(logging.CRITICAL)
    logging.getLogger("botocore").setLevel(logging.CRITICAL)
    logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)


class DummyLogger(object):
    """
    A dummy logger that does nothing.

    Useful for when you want to disable logging
    """

    def __getattr__(self, name):
        return lambda *args, **kwargs: None
