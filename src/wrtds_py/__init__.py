from importlib.metadata import version

from wrtds_py.models.classic import WRTDS
from wrtds_py.models.kalman import WRTDSK

__version__ = version("wrtds-py")

__all__ = ["WRTDS", "WRTDSK", "__version__"]
