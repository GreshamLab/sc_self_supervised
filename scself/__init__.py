__version__ = '0.3.0'

from ._mcv import (
    mcv,
    mcv_r2_per_cell
)
from ._noise2self import (
    noise2self,
    multimodal_noise2self
)
from .scaling import (
    TruncRobustScaler,
    TruncStandardScaler
)
from .utils.dot_product import (
    dot,
    sparse_dot_patch
)
