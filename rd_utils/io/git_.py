"""
This file contains a git utilities.

This is an exact copy the utils found in automation/app/utils/git_utils.py The reason they are duplicated is to ensure
decoupling of automation and src as we would create a separate repo for automation.
"""

import git
from beartype import beartype


@beartype
def get_current_local_commit(length: int = 12) -> str:
    """
    Must be called from a git repository.

    Args:
        length (int, optional): number of leading chars of the sha to return. Defaults to 7.

    Returns:
        str: commit sha
    """
    commit = git.Repo(search_parent_directories=True).commit().__str__()
    return commit[:length]


@beartype
def get_current_local_branch() -> str:
    """
    Must be called from a git repository - returns the current commit of the working branch

    Returns:
        str: name of the current branch
    """
    branch = git.Repo(search_parent_directories=True).active_branch.__str__()
    return branch
