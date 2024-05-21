# ruff: noqa: E402, F403
# make sure warnings are imported
import warnings

from decalmlutils.io.misc import *  # noqa: F401, F403
from decalmlutils.logging_utils import setup_default_logging

# always show deprecation warnings
warnings.simplefilter("always", DeprecationWarning)


__version__ = "0.0.4"


setup_default_logging()
