from .logging import log, verbose
from .dot_product import (
    dot,
    sparse_dot_patch
)
from .standardization import standardize_data
from .sum import array_sum
from .pairwise_loss import (
    pairwise_metric,
    mcv_mean_error,
    coefficient_of_variation,
    variance
)
from ._pca import pca
from .shrink import shrink_array_to_zero_inplace
from .cast_dtype_inplace import cast_to_float_inplace
