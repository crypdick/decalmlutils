import json
import logging

from beartype import beartype
from beartype.typing import Dict, Optional, Tuple

from mltoolkit.conf import settings
from mltoolkit.io.aws import get_aws_client
from mltoolkit.tensors import get_chunks

logger = logging.getLogger(__name__)


@beartype
def put_events(events: list[Tuple[str, Dict]]) -> Optional[list[Dict]]:
    """
    Wrapper of boto3 EventBridge client put_events.

    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/events.html#EventBridge.Client.put_events
    """

    bus_events = [
        {
            "Source": settings.AWS_EVENT_BUS_SOURCE,
            "DetailType": event_type,
            "Detail": json.dumps(payload),
            "EventBusName": settings.AWS_EVENT_BUS_ARN,
        }
        for event_type, payload in events
    ]

    try:
        client = get_aws_client("events")
        responses = []

        # eventbridge put_events has a limit of 10 events per request
        # https://docs.aws.amazon.com/eventbridge/latest/APIReference/API_PutEvents.html
        chunks = get_chunks(num_items=len(bus_events), chunk_size=10)
        for chunk in chunks:
            resp = client.put_events(Entries=bus_events[chunk[0] : chunk[-1]])
            responses.append(resp)

        return responses
    except Exception as e:
        logger.error(
            f"Failed to send event: {bus_events} with error: {e}", exc_info=True
        )
