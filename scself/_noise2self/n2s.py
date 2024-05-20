import numpy as np
import tqdm

from scself.utils import log, pca
from .graph import (
    neighbor_graph,
    local_optimal_knn
)
from .common import (
    _check_args,
    _standardize,
    _search_k
)


def noise2self(
    count_data,
    neighbors=None,
    npcs=None,
    metric='euclidean',
    loss='mse',
    loss_kwargs={},
    return_errors=False,
    connectivity=False,
    standardization_method='log',
    pc_data=None,
    chunk_size=10000,
    random_state=None,
    verbose=None
):
    """
    Select an optimal set of graph parameters based on noise2self

    :param count_data: Count data
    :type count_data: np.ndarray, sp.spmatrix
    :param neighbors: k values to search for global graph,
        defaults to None (5 to 105 by 10s)
    :type neighbors: np.ndarray, optional
    :param npcs: Number of PCs to use to embed graph,
        defaults to None (5 to 105 by 10s)
    :type npcs: np.ndarray, optional
    :param verbose: Verbose output to stdout,
        defaults to False
    :type verbose: bool, optional
    :param metric: Distance metric to use for k-NN graph construction,
        supports any metric sklearn Neighbors does, defaults to 'euclidean'
    :type metric: str, optional
    :param loss: Loss function for comparing reconstructed expression data to
        actual expression data, supports `mse`, `mae`, and `log_loss`, or any
        callable of the form func(x, y, **kwargs). Defaults to `mse`.
    :type loss: str, func, optional
    :param loss_kwargs: Dict of kwargs for the loss function, defalts to {}
    :type loss_kwargs: dict, optional
    :param return_errors: Return the mean square errors for global
        neighbor/nPC search, defaults to False
    :type return_errors: bool, optional
    :param use_sparse: Deprecated; always keep sparse now
    :type use_sparse: bool
    :param connectivity: Calculate row stochastic matrix from connectivity,
        not from distance
    :type connectivity: bool
    :param standardization_method: How to standardize provided count data,
        None disables. Options are `log`, `scale`, and `log_scale`. Defaults
        to 'log'.
    :type standardization_method: str, optional,
    :param pc_data: Precalculated principal components, defaults to None
    :type pc_data: np.ndarray, optional
    :return: Optimal k-NN graph
        global optimal # of PCs,
        global optimal k,
        local optimal k for each observation
    :rtype: sp.sparse.csr_matrix, int, int, np.ndarray [int]
    """

    neighbors, npcs = _check_args(
        neighbors,
        npcs,
        count_data,
        pc_data
    )
    _max_neighbors = np.max(neighbors)

    data_obj, expr_data = _standardize(
        count_data,
        standardization_method
    )

    log(
        f"Searching {len(npcs)} PCs [{np.min(npcs)}-{np.max(npcs)}] by "
        f"{len(neighbors)} Neighbors [{np.min(neighbors)}-{_max_neighbors}]"
    )

    if pc_data is not None:
        log(f"Using existing PCs {pc_data.shape}")
        data_obj.obsm['X_pca'] = pc_data

    else:
        data_obj.obsm['X_pca'] = pca(expr_data, np.max(npcs))

    mses = np.zeros((len(npcs), len(neighbors)))

    if len(npcs) > 1:

        # Search for the smallest MSE for each n_pcs / k combination
        # Outer loop does PCs, because the distance graph has to be
        # recalculated when PCs changes
        for i, pc in (pbar := tqdm.tqdm(enumerate(npcs), total=len(npcs))):

            pbar.set_description(f"{pc} PCs")

            # Calculate neighbor graph with the max number of neighbors
            # Faster to select only a subset of edges than to recalculate
            # (obviously)
            neighbor_graph(
                data_obj,
                pc,
                _max_neighbors,
                metric=metric,
                random_state=random_state
            )

            # Search through the neighbors space
            mses[i, :] = _search_k(
                expr_data,
                (data_obj.obsp['distances'], ),
                neighbors,
                connectivity=connectivity,
                loss=loss,
                loss_kwargs=loss_kwargs,
                chunk_size=chunk_size
            )

        # Get the index of the optimal PCs and k based on
        # minimizing MSE
        op_pc = np.argmin(np.min(mses, axis=1))
        op_k = np.argmin(mses[op_pc, :])

        log(
            f"Global optimal graph at {npcs[op_pc]} PCs "
            f"and {neighbors[op_k]} neighbors",
        )
    else:
        log(
            "Skipping global optimal graph search and "
            f"using {npcs[0]} PCs",
        )
        op_pc = 0
        op_k = None

    # Recalculate a k-NN graph from the optimal # of PCs
    neighbor_graph(
        data_obj,
        npcs[op_pc],
        _max_neighbors,
        metric=metric,
        random_state=random_state
    )

    # Search space for k-neighbors
    local_neighbors = np.arange(
        np.min(neighbors) if len(neighbors) > 1 else 1,
        _max_neighbors + 1
    )

    log(f"Searching local k ({local_neighbors[0]}-{local_neighbors[-1]})")

    # Search for the optimal number of k for each obs
    # For the global optimal n_pc
    local_error = _search_k(
        expr_data,
        (data_obj.obsp['distances'], ),
        local_neighbors,
        by_row=True,
        connectivity=connectivity,
        loss=loss,
        loss_kwargs=loss_kwargs,
        chunk_size=chunk_size,
        pbar=True
    )

    local_k = local_neighbors[np.argmin(local_error, axis=0)]

    log(
        f"Optimal local k complete ({local_k.min()}-{local_k.max()}, "
        f"mean={local_k.mean():.2f})"
    )

    # Pack return object:
    # Optimal (variable-k) graph
    # Optimal # of PCs
    # Optimal # of neighbors (global)
    # Optimal # of neighbors (local)
    optimals = (
        local_optimal_knn(
            data_obj.obsp['distances'],
            local_k
        ),
        npcs[op_pc],
        neighbors[op_k] if op_k is not None else None,
        local_k
    )

    if return_errors:
        return optimals, (mses, local_error)

    else:
        return optimals
