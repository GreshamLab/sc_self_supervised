import numpy as np
import scipy.sparse as sps
import anndata as ad

from scself.utils import (
    pairwise_metric,
    mcv_mean_error,
    array_sum
)


def molecular_split(
    count_data,
    random_seed=800,
    p=0.5
):
    """
    Break an integer count matrix into two count matrices.
    These will sum to the original count matrix and are
    selected randomly from the binomial distribution

    :param count_data: Integer count data
    :type count_data: np.ndarray, sp.sparse.csr_matrix, sp.sparse.csc_matrix
    :param random_seed: Random seed for generator, defaults to 800
    :type random_seed: int, optional
    :param p: Split probability, defaults to 0.5
    :type p: float, optional
    :return: Two count matrices A & B of the same type as the input count_data,
        where A + B = count_data
    :rtype: np.ndarray or sp.sparse.csr_matrix or sp.sparse.csc_matrix
    """

    rng = np.random.default_rng(random_seed)

    if sps.issparse(count_data):

        normalization_depth = np.median(
            array_sum(count_data, axis=1)
        )

        if sps.isspmatrix_csr(count_data):
            mat_func = sps.csr_matrix
        else:
            mat_func = sps.csc_matrix

        cv_data = mat_func((
            rng.binomial(count_data.data, p=p),
            count_data.indices,
            count_data.indptr),
            shape=count_data.shape
        )

        count_data = mat_func((
            count_data.data - cv_data.data,
            count_data.indices,
            count_data.indptr),
            shape=count_data.shape
        )

    else:

        normalization_depth = np.median(
            count_data.sum(axis=1)
        )

        cv_data = np.zeros_like(count_data)

        for i in range(count_data.shape[0]):
            cv_data[i, :] = rng.binomial(count_data[i, :], p=p)

        count_data = count_data - cv_data

    count_data = ad.AnnData(count_data)
    cv_data = ad.AnnData(cv_data)

    return count_data, cv_data, normalization_depth


def mcv_comp(
    x,
    pc,
    rotation,
    metric,
    calculate_r2=False,
    tss=None,
    axis=1,
    **metric_kwargs
):

    if metric == 'mse':
        metric_arr = mcv_mean_error(
            x,
            pc,
            rotation,
            axis=axis,
            squared=True,
            **metric_kwargs
        )
    elif metric == 'mae':
        metric_arr = mcv_mean_error(
            x,
            pc,
            rotation,
            axis=axis,
            squared=False,
            **metric_kwargs
        )
    else:
        metric_arr = pairwise_metric(
            x,
            pc @ rotation,
            metric=metric,
            axis=axis,
            **metric_kwargs
        )

    if calculate_r2:
        if tss is None:
            tss = array_sum(x, axis=axis, squared=True)
            tss = tss / x.shape[axis]

        if metric != 'mse':
            r2_array = mcv_mean_error(
                x,
                pc,
                rotation,
                axis=axis,
                squared=True
            )
        else:
            r2_array = metric_arr.copy()

        np.divide(
            r2_array,
            tss,
            where=tss != 0,
            out=r2_array
        )

        r2_array *= -1
        r2_array += 1
        r2_array[tss == 0] = 0.

        return metric_arr, r2_array

    else:
        return metric_arr
