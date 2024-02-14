# ruff: noqa: E402, F403
# make sure warnings are imported
import warnings

# always show deprecation warnings
warnings.simplefilter("always", DeprecationWarning)

# FIXME: replace blob with explicit imports in the future and re-enable F403
from .tensors import *

from .web import *

from .plot_functions import *

from .text import *

__version__ = "0.0.1"
