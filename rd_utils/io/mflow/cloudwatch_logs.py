import logging
import os
import sys
import traceback

import watchtower
from beartype import beartype
from beartype.typing import Optional

from mltoolkit.conf import settings


class CloudwatchHandlerError(Exception):
    pass


@beartype
def get_cloudwatch_logger(
    logger_name: str, run_id: str, strict: bool = False
) -> logging.Logger:
    """
    Sets up a logger with a Cloudwatch handler.

    Will disable Cloudwatch logging if mode is not prod.


    Args:
        logger_name: the name of the logger
        strict: if True, will raise an error adding the Cloudwatch handler fails.

    Raises:
        CloudwatchHandlerError: if the Cloudwatch handler cannot be added

    Returns:
        a logger with Cloudwatch handler
    """
    # do not overwrite `logger` in outer scope
    cloudwatch_logger = logging.getLogger(logger_name)
    """
    CloudWatch logger setup.
    """
    mode = os.environ.get("MODE", "prod")
    if mode == "prod":  # if prod, log to CloudWatch
        try:
            cloudwatch_logger.addHandler(
                watchtower.CloudWatchLogHandler(
                    log_group=settings.CLOUDWATCH_LOG_GROUP,
                    stream_name=run_id,
                    create_log_stream=True,
                    create_log_group=False,
                )
            )
        except Exception as e:
            if strict:
                raise CloudwatchHandlerError("Could not add Cloudwatch handler.") from e

    def excepthook(exc_type, exc_message, stacktrace):
        """
        A logger exception hook for logging tracebacks and exceptions.

        Source: https://stackoverflow.com/a/9929970/4212158
        """
        cloudwatch_logger.error(
            f"Logging an uncaught exception: {exc_message}. Traceback: \n{traceback.format_tb(stacktrace)}",
            exc_info=(exc_type, exc_message, stacktrace),
        )

    sys.excepthook = excepthook  # loggers will log exceptions

    return cloudwatch_logger


@beartype
def get_cloudwatch_log_link(run_id: Optional[str] = None) -> str:
    """
    Returns a link to the Cloudwatch log for the current run.

    Args:
        run_id: the run id

    Returns:
        a link to the Cloudwatch log for the current run
    """

    cloudwatch_template = "https://{AWS_REGION}.console.aws.amazon.com/cloudwatch/home?region={AWS_REGION}#logsV2:log-groups/log-group/{LOG_GROUP_NAME}/log-events/{LOG_STREAM_NAME}"
    log_url = cloudwatch_template.format(
        AWS_REGION=settings.AWS_REGION,
        LOG_GROUP_NAME=settings.CLOUDWATCH_LOG_GROUP,
        LOG_STREAM_NAME=run_id,
    )
    # Cloudwatch clobbers our link template (it replaces the last `/` with `$252F``).
    # Thus, in order to match the Cloudwatch log link, we need to clobber our link as well.
    log_url = "$252F".join(log_url.rsplit("/", 1))

    return log_url
