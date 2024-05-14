import tqdm
import numpy as np
import scipy.sparse as sps

from scself.utils import (
    standardize_data,
    log,
    pca
)
from .common import (
    mcv_comp,
    molecular_split
)


def mcv(
    count_data,
    n=1,
    n_pcs=100,
    random_seed=800,
    p=0.5,
    metric='mse',
    standardization_method='log',
    metric_kwargs={},
    silent=False,
    verbose=None,
    zero_center=False
):
    """
    Calculate a loss metric based on molecular crossvalidation

    :param count_data: Integer count data
    :type count_data: np.ndarray, sp.sparse.csr_matrix, sp.sparse.csc_matrix
    :param n: Number of crossvalidation resplits, defaults to 5
    :type n: int, optional
    :param n_pcs: Number of PCs to search, defaults to 100
    :type n_pcs: int, optional
    :param random_seed: Random seed for split, defaults to 800
    :type random_seed: int, optional
    :param p: Split probability, defaults to 0.5
    :type p: float, optional
    :param metric: Metric to use - accepts 'mse', 'mae', and 'r2' as strings,
        or any callable of the type metric(pred, true), defaults to 'mse'
    :type metric: str, func, optional
    :return: An n x n_pcs array of metric results
    :rtype: np.ndarray
    """

    n_pcs = min(n_pcs, *map(lambda x: x - 1, count_data.shape))
    size = count_data.shape[0] * count_data.shape[1]

    metric_arr = np.zeros((n, n_pcs + 1), dtype=float)

    if silent:
        level = 10
    else:
        level = 30

    # Use a single progress bar for nested loop
    for i in range(n):

        log(
            f"Iter #{i}: Splitting data {count_data.shape}",
            level=level
        )

        A, B, n_counts = molecular_split(
            count_data,
            random_seed=random_seed,
            p=p
        )

        log(
            f"Iter #{i}: Standardizing Train ({standardization_method}) "
            f"{A.shape}",
            level=level
        )

        A = standardize_data(
            A,
            target_sum=n_counts,
            method=standardization_method
        )

        log(
            f"Iter #{i}: Standardizing Test ({standardization_method}) "
            f"{B.shape}",
            level=level
        )

        B = standardize_data(
            B,
            target_sum=n_counts,
            method=standardization_method
        )

        # Calculate PCA
        log(f"Iter #{i}: Initial PCA ({n_pcs} comps)")
        pca(A, n_pcs, zero_center=zero_center)

        # Null model (no PCs)

        log(f"Iter #{i}: 0 PCs", level=10)

        if sps.issparse(B.X):
            metric_arr[i, 0] = np.sum(B.X.data ** 2) / size
        else:
            metric_arr[i, 0] = np.sum(B.X ** 2) / size

        # Calculate metric for 1-n_pcs number of PCs
        for j in tqdm.trange(1, n_pcs + 1):
            log(f"Iter #{i}: {j} PCs", level=10)
            metric_arr[i, j] = mcv_comp(
                B.X,
                A.obsm['X_pca'][:, 0:j],
                A.varm['PCs'][:, 0:j].T,
                metric=metric,
                axis=None,
                tss=metric_arr[i, 0],
                **metric_kwargs
            )

    return metric_arr
