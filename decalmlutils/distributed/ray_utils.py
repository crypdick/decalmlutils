import logging

import ray
from beartype import beartype
from beartype.typing import Any, Dict
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_fixed

# do not use cloudwatch logging here. it does not serialize for Ray
logger = logging.getLogger(__name__)


@beartype
def ray_init_no_flakiness(disable_retries: bool = False) -> None:
    """
    Keep trying to ray.init(), even if Bitbucket CI machines lie about available RAM.

    This is a hack to flaky CI tests, which is caused by Bitbucket's CI machines lying to Ray about their
    available RAM (sometimes it reports that there is negative RAM available).

    During production runs, this function behaves the same as a simple ray.init().

    Args:
        disable_retries: whether to disable retries. Set to true in production runs so that this function behaves like
            ray.init().
    """
    NUM_SECS_DEADLINE = 180  # 21-10-11 120 secs was not enough -rd
    for attempt in Retrying(
        wait=wait_fixed(1),  # 1 sec between attempts
        reraise=True,
        stop=stop_after_attempt(1 if disable_retries else NUM_SECS_DEADLINE),
        retry=retry_if_exception_type(ValueError),
    ):
        with attempt:
            try:
                ray.init(_redis_max_memory=10**8)
            # ValueError: After taking into account object store and redis memory usage, the amount of memory on this
            # node available for tasks and actors (0.03 GB) is less than 0% of total
            except ValueError as e:
                logger.info(
                    "failed to start Ray, probably due to Bitbucket not giving us enough RAM. retrying step"
                )
                ray.shutdown()  # will have re-init error without this
                raise e


class RayObjectBag:
    """
    Stores objects in Ray distributed object storage.

    Note: if you get an item from the bag and then edit it, this will _NOT_ update the item in the bag! If you want to
    do this, you need to re-add the item to the bag (and also edit __setitem__ to permit this).
    """

    def __init__(self, **kwargs):
        self.object_refs: Dict[Any, ray.ObjectRef] = {}

        if kwargs:
            self.update(**kwargs)

    def __setitem__(self, key, value) -> None:
        assert (
            key not in self.object_refs.keys()
        ), f"Cannot replace {key}, objects are currently treated as immutable"
        self.object_refs[key] = ray.put(value)

    @beartype
    def __getitem__(self, item) -> Any:
        return ray.get(self.object_refs[item])

    def __repr__(self):
        return "bag"

    def keys(self) -> set:
        return set(self.object_refs.keys())

    @beartype
    def as_dict(self) -> Dict[Any, Any]:
        # convert object refs to objects before sending back
        return {k: ray.get(v) for k, v in self.object_refs.items()}

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self[k] = v  # will be ray.put() in other function
