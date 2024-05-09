from .logging import log, verbose
from .dot_product import (
    dot,
    sparse_dot_patch
)
from .standardization import standardize_data
from .sum import array_sum
from .pairwise_loss import (
    pairwise_metric,
    mcv_mse,
    coefficient_of_variation,
    variance
)
