import scanpy as sc

from scself.sparse import is_csr
from scself.scaling import TruncRobustScaler


def _normalize_for_pca(
    count_data,
    target_sum=None,
    log=False,
    scale=False
):
    """
    Depth normalize and log pseudocount
    This operation will be entirely inplace

    :param count_data: Integer data
    :type count_data: ad.AnnData
    :return: Standardized data
    :rtype: np.ad.AnnData
    """

    if is_csr(count_data.X):
        from ..sparse.math import sparse_normalize_total
        sparse_normalize_total(
            count_data.X,
            target_sum=target_sum
        )

    else:
        sc.pp.normalize_total(
            count_data,
            target_sum=target_sum
        )

    if log:
        sc.pp.log1p(count_data)

    if scale:
        scaler = TruncRobustScaler(with_centering=False)
        scaler.fit(count_data.X)

        if is_csr(count_data.X):
            from ..sparse.math import sparse_normalize_columns
            sparse_normalize_columns(
                count_data.X,
                scaler.scale_
            )
        else:
            count_data.X = scaler.transform(
                count_data.X
            )

    return count_data


def standardize_data(
    count_data,
    target_sum=None,
    method='log'
):

    if method == 'log':
        return _normalize_for_pca(
            count_data,
            target_sum,
            log=True
        )
    elif method == 'scale':
        return _normalize_for_pca(
            count_data,
            target_sum,
            scale=True
        )
    elif method == 'log_scale':
        return _normalize_for_pca(
            count_data,
            target_sum,
            log=True,
            scale=True
        )
    elif method == 'depth':
        return _normalize_for_pca(
            count_data,
            target_sum
        )
    elif method is None:
        return count_data
    else:
        raise ValueError(
            'method must be None, `depth`, `log`, `scale`, or `log_scale`, '
            f'{method} provided'
        )
