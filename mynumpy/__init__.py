from .core import *
from .lib.scimath import *
from .lib.twodim_base import *
from .version import __version__


def __getattr__(attr):
    if attr == "linalg":
        import mynumpy.linalg as linalg

        return linalg
