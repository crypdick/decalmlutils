# ruff: noqa: E402, F403
# make sure warnings are imported
import warnings

from .misc import *  # noqa: F401, F403

# always show deprecation warnings
warnings.simplefilter("always", DeprecationWarning)


__version__ = "0.0.11"
