import numpy as np

from scself.utils import (
    standardize_data,
    log,
    pca
)
from .common import (
    mcv_comp,
    molecular_split
)


def mcv_r2_per_cell(
    count_data,
    n=5,
    n_pcs=100,
    random_seed=800,
    p=0.5,
    standardization_method='log',
    zero_center=False,
    silent=False
):
    """
    Calculate r2 per cell based on molecular crossvalidation

    :param count_data: Integer count data
    :type count_data: np.ndarray, sp.sparse.csr_matrix, sp.sparse.csc_matrix
    :param n: Number of crossvalidation resplits, defaults to 5
    :type n: int, optional
    :param n_pcs: Number of PCs to evaluate, defaults to 100
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
    n_obs = count_data.shape[0]

    metric_arr = np.zeros((n_obs, n), dtype=float)
    r2_arr = np.zeros((n_obs, n), dtype=float)

    if silent:
        level = 10
    else:
        level = 30

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
        log(f"Iter #{i}: Initial PCA ({n_pcs} comps)", level=level)
        pca(A, n_pcs, zero_center=zero_center)

        # Null model (no PCs)

        log(f"Iter #{i}: Calculating Metrics", level=level)

        metric_arr[:, i], r2_arr[:, i] = mcv_comp(
            B.X,
            A.obsm['X_pca'],
            A.varm['PCs'].T,
            metric='mse',
            axis=1,
            calculate_r2=True,
        )

    return metric_arr, r2_arr
