import numpy as np


def cast_to_float_inplace(arr):

    if arr.dtype == np.int32:
        dtype = np.float32
    elif arr.dtype == np.int64:
        dtype = np.float64
    else:
        dtype = None

    if dtype is not None:
        float_view = arr.view(dtype)
        float_view[:] = arr
        arr.dtype = dtype

    return arr
