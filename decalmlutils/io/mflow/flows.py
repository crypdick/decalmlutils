import logging
import os

from beartype import beartype
from beartype.typing import Optional

logger = logging.getLogger(
    __name__
)  # do not use cloudwatch logging here due to circular imports

METAFLOW_PROD_PROFILE = "prod"
UPSTREAM_PIPELINE_LINKING_PREFIX = "srcrun_"


@beartype
def comma_separated_to_list(comma_separated_list: Optional[str]) -> list[str]:
    """
    Converts a comma seperated list to list of strings or empty list if string is empty.
    """
    if not comma_separated_list:
        return []
    return comma_separated_list.split(",")


@beartype
def get_metaflow_profile() -> str:
    """
    Instead of assigning the value of METAFLOW_PROFILE to a constant 'MODE' which can be inadvertently reset, wrap in a
    function to effectively make read-only.
    """
    return os.environ.get("METAFLOW_PROFILE", METAFLOW_PROD_PROFILE)


@beartype
def is_test() -> bool:
    """
    Use a function to prevent typos in 'test' string during this check.
    """
    return get_metaflow_profile() == "test"
