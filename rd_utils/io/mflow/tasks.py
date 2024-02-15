from beartype import beartype
from beartype.typing import Literal, Optional
from metaflow import Run, Task


@beartype
def get_all_tasks(run: Run) -> list[Task]:
    """
    Returns all the tasks of a metaflow run.

    Args:
        run (Run): Metaflow Run

    Returns:
        list[Task]: Metaflow Task
    """
    tasks = []
    for step in run:
        tasks.extend(step.tasks())
    return tasks


class NoTasksFoundError(Exception):
    pass


@beartype
def get_last_finished_task(
    run: Run, no_steps: Literal["any", "raise"] = "raise"
) -> Optional[Task]:
    """
    Get the last finished step of a flow.

    Args:
        run (Run): Metaflow Run
        no_steps (Literal): options in the case where no steps are finished. "any" will typically return the start step,
            raise will raise an error

    Raises:
        ValueError: if no steps are finished and no_step option is set to raise

    Returns:
        Task: Metaflow Task
    """
    tasks = get_all_tasks(run)
    if len(tasks) == 0:
        if no_steps == "any":
            return None
        else:
            raise NoTasksFoundError(f"No tasks found for run {run.id}")
    latest_task_time = None
    latest_task = None
    for t in tasks:
        t_time = t.finished_at
        if latest_task_time is None or (
            t_time is not None and t_time > latest_task_time
        ):
            latest_task = t
            latest_task_time = t_time

    if no_steps == "any" or latest_task_time is not None or latest_task is not None:
        return latest_task
    elif no_steps == "raise":
        raise NoTasksFoundError(
            f"No steps have yet been completed for flow {run.id} with success status {run.successful} and `no_steps` "
            f"has been set to {no_steps}"
        )
