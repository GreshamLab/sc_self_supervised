import numpy as np
import tqdm


from scself.utils import log, pca
from .graph import (
    neighbor_graph,
    local_optimal_knn
)
from .common import (
    _check_args,
    _check_input_data,
    _standardize,
    _search_k
)


def multimodal_noise2self(
    count_data,
    target_data_index=0,
    standardization_method=None,
    neighbors=None,
    npcs=None,
    metric='euclidean',
    loss='mse',
    loss_kwargs={},
    return_errors=False,
    connectivity=False,
    pc_data=None,
    chunk_size=10000,
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
        npcs
    )

    if not isinstance(count_data, (list, tuple)):
        raise ValueError(
            f"count_data must be a list of count matrices; "
            f"{type(count_data)} provided"
        )

    _n_modes = len(count_data)

    if pc_data is None:
        pc_data = [None] * _n_modes
    elif not isinstance(pc_data, (list, tuple)):
        raise ValueError(
            f"pc_data must be a list of PC matrices; "
            f"{type(pc_data)} provided"
        )

    for c, p in zip(count_data, pc_data):
        _check_input_data(npcs, c, p)

    data_obj = [None] * _n_modes
    expr_data = [None] * _n_modes

    for i in range(_n_modes):

        data_obj[i], expr_data[i] = _standardize(
            count_data[i],
            standardization_method
        )

    log(f"Searching {len(npcs)} PC x {len(neighbors)} Neighbors space")

    for i in range(_n_modes):
        if pc_data[i] is not None:
            log(f"Using existing PCs {pc_data[i].shape}")
            data_obj[i].obsm['X_pca'] = pc_data[i]

        else:
            pca(data_obj[i], expr_data[i], np.max(npcs))

    mses = np.zeros((len(npcs), len(neighbors)))

    if len(npcs) > 1:

        # Search for the smallest MSE for each n_pcs / k combination
        # Outer loop does PCs, because the distance graph has to be
        # recalculated when PCs changes
        for i, pc in tqdm.tqdm(enumerate(npcs), total=len(npcs)):

            # Calculate neighbor graph with the max number of neighbors
            # Faster to select only a subset of edges than to recalculate
            # (obviously)
            for i in range(_n_modes):
                neighbor_graph(
                    data_obj[i],
                    pc,
                    np.max(neighbors),
                    metric=metric
                )

            # Search through the neighbors space
            mses[i, :] = _search_k(
                expr_data[target_data_index],
                [data_obj[i].obsp['distances'] for i in range(_n_modes)],
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
    for i in range(_n_modes):
        neighbor_graph(
            data_obj[i],
            npcs[op_pc],
            np.max(neighbors),
            metric=metric
        )

    # Search space for k-neighbors
    local_neighbors = np.arange(
        np.min(neighbors) if len(neighbors) > 1 else 1,
        np.max(neighbors)
    )

    # Search for the optimal number of k for each obs
    # For the global optimal n_pc
    local_k = local_neighbors[np.argmin(
        _search_k(
            expr_data[target_data_index],
            [data_obj[i].obsp['distances'] for i in range(_n_modes)],
            local_neighbors,
            by_row=True,
            connectivity=connectivity,
            loss=loss,
            loss_kwargs=loss_kwargs,
            chunk_size=chunk_size
        ),
        axis=0
    )]

    # Pack return object:
    # Optimal (variable-k) graphs
    # Optimal # of PCs
    # Optimal # of neighbors (global)
    # Optimal # of neighbors (local)
    optimals = (
        [
            local_optimal_knn(
                data_obj[i].obsp['distances'],
                local_k
            )
            for i in range(_n_modes)
        ],
        npcs[op_pc],
        neighbors[op_k] if op_k is not None else None,
        local_k
    )

    if return_errors:
        return optimals, mses

    else:
        return optimals
