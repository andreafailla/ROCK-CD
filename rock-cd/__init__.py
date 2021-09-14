import os
from .rockcd import RockCD

__version__ = "0.1"
__license__ = "GPLv3"
__authors__ = "Andrea Failla, Federico Mazzoni"
__email__ = (
    "andrea[dot]failla[dot]ak[at]gmail[dot]com, mazzoni[dot]federico1[at]gmail[dot]com"
)


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = ["RockCD"]

