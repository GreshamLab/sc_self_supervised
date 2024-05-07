import numpy as np
import scipy.sparse as sps
import anndata as ad
import tqdm

from scself.utils import (
    dot,
    standardize_data,
    pairwise_metric
)
from scself.sparse import is_csr
from scself.utils.dot_product import sparse_dot_patch
from .graph import (
    local_optimal_knn,
    _connect_to_row_stochastic,
    _dist_to_row_stochastic
)

N_PCS = np.arange(5, 115, 10)
N_NEIGHBORS = np.arange(15, 115, 10)


def _search_k(
    X,
    graph,
    k,
    by_row=False,
    loss='mse',
    loss_kwargs={},
    X_compare=None,
    pbar=False,
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

    X_compare = X_compare if X_compare is not None else X

    mses = np.zeros(n_k) if not by_row else np.zeros((n_k, n))

    rfunc = tqdm.trange if pbar is True else range

    if hasattr(pbar, 'postfix'):
        _postfix = pbar.postfix
    else:
        _postfix = None

    if connectivity:
        row_normalize = _connect_to_row_stochastic
    else:
        row_normalize = _dist_to_row_stochastic

    for i in rfunc(n_k):

        if _postfix is not None:
            pbar.postfix = _postfix + f" ({k[i]} N)"
            pbar.update(1)

        # Extract k non-zero neighbors from the graph
        k_graph = local_optimal_knn(
            graph.copy(),
            np.full(n, k[i]),
            keep='smallest'
        )

        # Convert to a row stochastic graph
        k_graph = row_normalize(k_graph)

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

        if by_row:
            return _row_mse
        else:
            return np.mean(_row_mse)

    else:
        return pairwise_metric(
            X,
            dot(k_graph, X, dense=not sps.issparse(X)),
            by_row=by_row,
            metric=metric,
            **loss_kwargs
        )


def _check_args(
    neighbors,
    npcs,
    count_data,
    pc_data
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

    _max_pcs = np.max(npcs)

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

    # Check input data sizes
    if pc_data is not None and pc_data.shape[1] < _max_pcs:
        raise ValueError(
            f"Cannot search through {_max_pcs} PCs; only "
            f"{pc_data.shape[1]} components provided"
        )
    elif min(count_data.shape) < _max_pcs:
        raise ValueError(
            f"Cannot search through {_max_pcs} PCs for "
            f"data {count_data.shape} provided"
        )

    return neighbors, npcs


def _standardize(count_data, standardization_method):
    # Standardize data if necessary and create an anndata object
    # Keep separate reference to expression data and force float32
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
