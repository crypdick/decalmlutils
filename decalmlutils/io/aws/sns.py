import json
import logging

from beartype import beartype

from decalmlutils.conf import settings
from decalmlutils.io.aws import get_aws_client

logger = logging.getLogger(__name__)


@beartype
def publish_to_sns(msg: str, subject: str, verbose: bool = True) -> None:
    client = get_aws_client("sns")
    if verbose:
        logger.info(f"Publishing to SNS topic: {settings.SNS_TOPIC_ARN}")
    client.publish(
        TopicArn=settings.SNS_TOPIC_ARN,
        # PhoneNumber='',
        Message=json.dumps({"default": msg}),
        Subject=subject,
        MessageStructure="json",
    )

    if verbose:
        logger.info("Finished publishing to SNS topic")
