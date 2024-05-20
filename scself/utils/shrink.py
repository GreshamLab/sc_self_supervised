import numba
import numpy as np


@numba.njit(parallel=True)
def shrink_array_to_zero_inplace(
    arr,
    shrink_value
):

    n = arr.shape[0]

    if arr.ndim == 1:
        for i in numba.prange(n):
            if abs(arr[i]) < shrink_value:
                arr[i] = 0

    else:
        for i in numba.prange(n):
            arr[i][np.abs(arr[i]) < shrink_value] = 0
