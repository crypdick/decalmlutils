import datetime
import logging
import time

from beartype import beartype
from beartype.typing import Optional
from metaflow import Flow, Run, current

from decalmlutils.mflow.runs import get_run_pathspec

logger = logging.getLogger(
    __name__
)  # do not use cloudwatch logging here due to circular imports


@beartype
def wait_for_recent_runs_to_finish(
    flow: Optional[str] = None,
    current_run_pathspec: Optional[str] = None,
    max_wait_per_recent_run: int = 120,
) -> int:
    """
    This method looks back to the most recent `runs_to_check` number of runs of a given flow (or current flow). It then
    enforces a sleep of `max_wait_per_recent_run` seconds for each recent ran that either has not finished or until all
    recent runs have finished, whichever happens first.

    Args:
        flow (Optional[str], optional): flow name. Defaults to current flow's name.
        current_run_pathspec (Optional[str], optional): run pathspec to look back from. Defaults to current run's pathspec.
        max_wait_per_recent_run (int, optional): max seconds to wait per run if they don't finish. Defaults to 60.

    Returns:
        int: rough number of seconds waited (slight underestimation as its based on counting not actual timing)
    """
    flow = Flow(flow or current.flow_name)
    current_run_pathspec = current_run_pathspec or get_run_pathspec()
    curr_run = Run(str(current_run_pathspec))
    curr_created_at = curr_run.created_at

    recency_window = datetime.timedelta(seconds=max_wait_per_recent_run)

    logger.info(f"Review previous runs for {flow}")

    recent_unfinished_runs = []
    for run in flow.runs():
        run: Run = run
        run_created_at = run.created_at

        # only look at older runs
        if curr_created_at <= run_created_at:
            logger.info(
                f"Skipping {run} as it was not before the current run {current_run_pathspec}"
            )

        # if unfinished or recent
        elif run.finished:
            logger.info(f"Skipping {run} as it is already finished")

        # collect all recent runs
        elif curr_created_at - run.created_at < recency_window:
            logger.info(f"{run} is still running")
            recent_unfinished_runs.append(run)

        # break out when the first non-recent run appears as list of runs are ordered by recency anyway so
        # the rest will all be older
        else:
            break

    waited_seconds = 0
    while (
        (len(recent_unfinished_runs) * max_wait_per_recent_run) - waited_seconds
    ) > 0:
        logger.info(
            f"Unfinished previous runs remaining: {recent_unfinished_runs} - Sleeping for 15 seconds..."
        )
        time.sleep(15)
        waited_seconds += 15
        recent_unfinished_runs = [
            rn for rn in recent_unfinished_runs if not rn.finished
        ]

    return waited_seconds
