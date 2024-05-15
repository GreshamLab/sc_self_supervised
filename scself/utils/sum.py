import numpy as np
import scipy.sparse as sps
from scself.sparse import is_csr


def array_sum(array, axis=None, squared=False):

    if not sps.issparse(array):
        return array.sum(axis)

    # If it's not a CSR array use builtins
    elif not is_csr(array):

        if squared and not sps.issparse(array):
            _sums = (array ** 2).sum(axis=axis)
        elif squared:
            _sums = array.power(2).sum(axis=axis)
        else:
            _sums = array.sum(axis=axis)
        try:
            _sums = _sums.A1
        except AttributeError:
            pass
        return _sums

    # If it is but sum is flattened sum over data
    if axis is None:
        return np.sum(array.data)

    # Otherwise go to numba functions
    # Because the scipy sparse method here is
    # very memory inefficient
    else:
        from ..sparse.math import sparse_sum

        return sparse_sum(
            array,
            axis=axis,
            squared=squared
        )
