import numpy as np
import scipy.sparse as sps

from pynndescent import PyNNDescentTransformer
from sklearn.neighbors import NearestNeighbors

from scself.sparse import is_csr
from scself.sparse.graph import _shrink_sparse_graph_k
from scself.utils import array_sum, dot


def local_optimal_knn(
    neighbor_graph,
    nn_vector,
    keep='smallest'
):
    """
    Modify a k-NN graph *in place* to have a specific number of
    non-zero values k per row based on a vector of k-values

    :param neighbor_graph: N x N matrix with edge values.
    :type neighbor_graph: np.ndarray, sp.sparse.csr_matrix
    :param nn_vector: Vector of `k` values per row
    :type nn_vector: np.ndarray
    :param keep: Keep the 'largest' or 'smallest' non-zero values
    :type keep: str
    :raise ValueError: Raise a ValueError if neighbor_graph is a
        non-CSR sparse matrix or if keep is not 'smallest' or
        'largest'
    :return: Return a reference to neighbor_graph
    :rtype: np.ndarray, sp.sparse.csr_matrix
    """

    n, m = neighbor_graph.shape

    neighbor_sparse = sps.issparse(neighbor_graph)

    if neighbor_sparse and not is_csr(neighbor_graph):
        raise ValueError("Sparse matrices must be CSR")

    if n != len(nn_vector):
        raise ValueError(
            f"{len(nn_vector)}-length vector wrong size "
            f" for graph {neighbor_graph.shape}"
        )

    # Define functions that select the inverse of
    # the k largest or smallest values
    # So those values can be zeroed
    if keep == 'smallest':
        def _nn_slice(k, n):
            return slice(k, None)

    elif keep == 'largest':
        def _nn_slice(k, n):
            return slice(None, n - k)
    else:
        raise ValueError("keep must be 'smallest' or 'largest'")

    _smallest = keep == 'smallest'

    if neighbor_sparse:
        _shrink_sparse_graph_k(
            neighbor_graph.data,
            neighbor_graph.indptr,
            np.asanyarray(nn_vector),
            smallest=_smallest
        )

        neighbor_graph.eliminate_zeros()

        return neighbor_graph

    for i in range(n):

        k = nn_vector[i]

        if k >= n:
            continue

        n_slice = neighbor_graph[i, :]

        # Use a masked array to block out zeros
        droppers = np.ma.masked_array(
            n_slice,
            mask=n_slice == 0
        ).argsort(endwith=_smallest)[_nn_slice(k, m)]

        # Write the new data into the original array
        neighbor_graph[i, droppers] = 0.

    return neighbor_graph


def set_diag(X, diag):
    """
    Set diagonal of matrix X.
    Safe for dense or sparse matrices

    :param X: Numeric matrix
    :type X: np.ndarray, sp.spmatrix
    :param diag: Value to fill
    :type diag: Numeric
    :return: Numeric matrix with set diagonal
    :rtype: np.ndarray, sp.spmatrix
    """

    if diag is None:
        pass

    elif sps.issparse(X):
        # UGLY HACK TO AVOID SPARSITY CHANGES #
        _remove_diag = X.diagonal()

        if np.all(_remove_diag == 0.):
            return X

        _remove_idx = np.arange(len(_remove_diag))

        X -= sps.csr_matrix(
            (_remove_diag, (_remove_idx, _remove_idx)),
            shape=X.shape
        )

    else:
        np.fill_diagonal(X, diag)

    return X


def neighbor_graph(adata, pc, k, metric='euclidean'):
    """
    Build neighbor graph in an AnnData object
    """

    if adata.n_obs < 25000:

        adata.obsp['distances'] = NearestNeighbors(
            n_neighbors=k,
            metric=metric,
            n_jobs=-1
        ).fit(adata.obsm['X_pca'][:, :pc]).kneighbors_graph()

    else:
        adata.obsp['distances'] = PyNNDescentTransformer(
            n_neighbors=k,
            n_jobs=None,
            metric=metric,
            n_trees=min(64, 5 + int(round((adata.n_obs) ** 0.5 / 20.0))),
            n_iters=max(5, int(round(np.log2(adata.n_obs)))),
        ).fit_transform(adata.obsm['X_pca'][:, :pc])

    # Enforce diagonal zeros on graph
    # Single precision floats
    set_diag(adata.obsp['distances'], 0)
    adata.obsp['distances'].data = adata.obsp['distances'].data.astype(
        np.float32
    )

    return adata


def _dist_to_row_stochastic(graph):

    if sps.isspmatrix(graph):

        rowsum = array_sum(graph, axis=1).astype(float)
        rowsum[rowsum == 0] = 1.

        # Dot product between inverse rowsum diagonalized
        # and graph.
        # Somehow faster then element-wise \_o_/

        if is_csr(graph):
            from ..sparse.math import _csr_row_divide

            _csr_row_divide(
                graph.data,
                graph.indptr,
                rowsum
            )
            return graph
        else:
            return dot(
                sps.diags(
                    (1. / rowsum),
                    offsets=0,
                    shape=graph.shape,
                    format='csr',
                    dtype=graph.dtype
                ),
                graph
            )
    else:

        rowsum = graph.sum(axis=1)
        rowsum[rowsum == 0] = 1.

        return np.multiply(
            graph,
            (1 / rowsum)[:, None],
            out=graph
        )


def _connect_to_row_stochastic(graph):

    graph_dtype = graph.dtype

    if sps.issparse(graph):
        graph.data[:] = 1
    else:
        graph = graph != 0
        graph = graph.astype(graph_dtype)

    return _dist_to_row_stochastic(graph)
