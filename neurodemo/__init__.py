from .neuronsim import *
from .runner import SimRunner
from .colormaps import *

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("neurodemo")
except PackageNotFoundError:
    __version__ = "unknown"
