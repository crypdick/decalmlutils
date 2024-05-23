# ruff: noqa: E402, F403
# make sure warnings are imported
import warnings

from .logging_utils import setup_default_logging
from .misc import *  # noqa: F401, F403

# always show deprecation warnings
warnings.simplefilter("always", DeprecationWarning)


__version__ = "0.0.8"


setup_default_logging()
