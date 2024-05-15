import numpy as np
import scipy.sparse as sps
import anndata as ad

from scself.utils import (
    dot,
    standardize_data,
    pairwise_metric,
    sparse_dot_patch
)
from scself.sparse import is_csr
from .graph import (
    local_optimal_knn,
    combine_row_stochastic_graphs,
    _connect_to_row_stochastic,
    _dist_to_row_stochastic,
    _invert_distance_graph
)

N_PCS = np.arange(5, 115, 10)
N_NEIGHBORS = np.arange(15, 115, 10)


def row_normalize(
    graph,
    connectivity=False,
    copy=False
):

    if copy:
        graph = graph.copy()

    if connectivity:
        return _connect_to_row_stochastic(graph)
    else:
        return _dist_to_row_stochastic(
            _invert_distance_graph(
                graph
            )
        )


def _search_k(
    X,
    graphs,
    k,
    by_row=False,
    loss='mse',
    loss_kwargs={},
    connectivity=False,
    chunk_size=10000
):
    """
    Find optimal number of neighbors for a given graph

    :param X: Data [M x N]
    :type X: np.ndarray, sp.spmatrix
    :param graph: Graph
    :type graph: np.ndarray, sp.spmatrix
    :param k: k values to search
    :type k: np.ndarray [K]
    :param by_row: Get optimal k for each observation,
        defaults to False
    :type by_row: bool, optional
    :param pbar: Show a progress bar, defaults to False
    :type pbar: bool
    :return: Mean Squared Error for each k [K] or
        for each k and each observation [K x M]
    :rtype: np.ndarray
    """

    n, _ = X.shape
    n_k = len(k)

    mses = np.zeros(n_k) if not by_row else np.zeros((n_k, n))

    for i in range(n_k):

        k_graph = [
            # Convert to a row stochastic graph
            row_normalize(
                # Extract k non-zero neighbors from the graph
                local_optimal_knn(
                    graph.copy(),
                    np.full(n, k[i]),
                    keep='smallest'
                ),
                connectivity=connectivity
            )
            for graph in graphs
        ]

        if len(k_graph) == 1:
            k_graph = k_graph[0]
        else:
            k_graph = combine_row_stochastic_graphs(k_graph)

        # Calculate mean squared error
        mses[i] = _noise_to_self_error(
            X,
            k_graph,
            by_row=by_row,
            metric=loss,
            chunk_size=chunk_size,
            **loss_kwargs
        )

    return mses


def _noise_to_self_error(
    X,
    k_graph,
    by_row=False,
    metric='mse',
    chunk_size=10000,
    **loss_kwargs
):

    if (metric == 'mse' and is_csr(X) and chunk_size is not None):

        from ..sparse.graph import _chunk_graph_mse

        _n_row = X.shape[0]
        _row_mse = np.zeros(X.shape[0], dtype=float)

        for i in range(int(np.ceil(_n_row / chunk_size))):
            _start = i * chunk_size
            _end = min(_start + chunk_size, _n_row)

            _row_mse[_start:_end] = _chunk_graph_mse(
                X,
                k_graph,
                _start,
                _end
            )

    else:
        _row_mse = pairwise_metric(
            X,
            dot(k_graph, X, dense=not sps.issparse(X)),
            metric=metric,
            **loss_kwargs
        )

    if by_row:
        return _row_mse
    else:
        return np.mean(_row_mse)


def _check_args(
    neighbors,
    npcs,
    count_data=None,
    pc_data=None
):

    # Get default search parameters and check dtypes
    if neighbors is None:
        neighbors = N_NEIGHBORS
    else:
        neighbors = np.asanyarray(neighbors).reshape(-1)

    if npcs is None:
        npcs = N_PCS
    else:
        npcs = np.asanyarray(npcs).reshape(-1)

    if not np.issubdtype(neighbors.dtype, np.integer):
        raise ValueError(
            "k-NN graph requires k to be integers; "
            f"{neighbors.dtype} provided"
        )

    if not np.issubdtype(npcs.dtype, np.integer):
        raise ValueError(
            "n_pcs must be integers; "
            f"{npcs.dtype} provided"
        )

    if count_data is not None:
        _check_input_data(npcs, count_data, pc_data)

    return neighbors, npcs


def _check_input_data(
    npcs,
    count_data,
    pc_data
):

    _max_pcs = np.max(npcs)

    if count_data is not None:
        _min_dim = min(count_data.shape)
    else:
        _min_dim = None

    # Check input data sizes
    if pc_data is not None and pc_data.shape[1] < _max_pcs:
        raise ValueError(
            f"Cannot search through {_max_pcs} PCs; only "
            f"{pc_data.shape[1]} components provided"
        )

    if _min_dim is not None and _min_dim < _max_pcs:
        raise ValueError(
            f"Cannot search through {_max_pcs} PCs for "
            f"data {count_data.shape} provided"
        )

    if count_data is not None and pc_data is not None:
        if count_data.shape[0] != pc_data.shape[0]:
            raise ValueError(
                f"Count data {count_data.shape} observations "
                f"do not match PC {pc_data.shape} observations"
            )


def _standardize(count_data, standardization_method):
    # Standardize data if necessary and create an anndata object
    # Keep separate reference to expression data and force float32
    # This way if the expression data is provided it isnt copied
    if standardization_method is not None:
        data_obj = standardize_data(
            ad.AnnData(count_data.astype(np.float32)),
            method=standardization_method
        )
        expr_data = data_obj.X

    else:
        data_obj = ad.AnnData(
            sps.csr_matrix(count_data.shape, dtype=np.float32)
        )
        expr_data = count_data.astype(np.float32)

    if sps.issparse(expr_data):
        sparse_dot_patch(expr_data)

    return data_obj, expr_data
