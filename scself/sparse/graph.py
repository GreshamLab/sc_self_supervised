import numba
import numpy as np
import scipy.sparse as sps


try:
    from sparse_dot_mkl import dot_product_mkl as dot

except ImportError:

    def dot(x, y, dense=False, cast=False, out=None):

        z = x @ y

        if dense and sps.issparse(z):
            z = z.A

        if out is not None:
            out[:] = z
            return out
        else:
            return z


@numba.njit(parallel=True)
def _shrink_sparse_graph_k(
    graph_data,
    graph_indptr,
    k_vec,
    smallest=True
):

    n = graph_indptr.shape[0] - 1

    for i in numba.prange(n):

        _left = graph_indptr[i]
        _right = graph_indptr[i+1]
        _n = _right - _left

        k = k_vec[i]

        if _n <= k:
            pass

        else:
            _data = graph_data[_left:_right]

            if smallest:
                _data[np.argsort(_data)[k - _n:]] = 0
            else:
                _data[np.argsort(_data)[:_n - k]] = 0


def _chunk_graph_mse(
    X,
    k_graph,
    row_start=0,
    row_end=None
):
    if row_end is None:
        row_end = k_graph.shape[0]
    else:
        row_end = min(k_graph.shape[0], row_end)

    return _mse_rowwise(
        X.data,
        X.indices,
        X.indptr,
        dot(
            k_graph[row_start:row_end, :],
            X,
            dense=True
        )
    )


@numba.njit(parallel=True)
def _mse_rowwise(
    a_data,
    a_indices,
    a_indptr,
    b
):

    n_row = b.shape[0]
    output = np.zeros(n_row, dtype=float)

    for i in numba.prange(n_row):

        _idx_a = a_indices[a_indptr[i]:a_indptr[i + 1]]
        _nnz_a = _idx_a.shape[0]

        row = b[i, :]

        if _nnz_a == 0:
            pass

        else:
            row = row.copy()
            row[_idx_a] -= a_data[a_indptr[i]:a_indptr[i + 1]]

        output[i] = np.mean(row ** 2)

    return output
