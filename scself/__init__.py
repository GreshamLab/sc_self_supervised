__version__ = "0.1.0"

from ._mcv.molecular_crossvalidation import mcv
from ._noise2self.n2s import noise2self
from .scaling import (
    TruncRobustScaler,
    TruncStandardScaler
)
from .utils.dot_product import (
    dot,
    sparse_dot_patch
)