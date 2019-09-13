
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .acquisition import *
from .bayesgpr import *
from .optimizer import *
from .utils import *