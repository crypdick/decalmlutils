"""
Utilities for testing.
"""

import importlib
import pkgutil
from types import ModuleType

from beartype import beartype
from beartype.typing import Dict, Union


@beartype
def _import_submodules(package: Union[str, ModuleType]) -> Dict[str, ModuleType]:
    """
    Import all submodules of a module, recursively, including subpackages.

    Useful for finding broken imports in a package in unit tests.

    https://stackoverflow.com/a/25562415/4212158
    """
    if isinstance(package, str):
        try:
            package = importlib.import_module(package)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"Could not import package: {package}") from e
    results = {}
    for _loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        try:
            results[full_name] = importlib.import_module(full_name)
            if is_pkg:
                results.update(_import_submodules(full_name))  # recurse into subpackage
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"Could not import module: {full_name}") from e
    return results
