import scanpy as sc

from scself.sparse import is_csr
from scself.scaling import TruncRobustScaler


def _normalize_for_pca(
    count_data,
    target_sum=None,
    log=False,
    scale=False,
    scale_factor=None
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

        if scale_factor is None:
            scaler.fit(count_data.X)
            scale_factor = scaler.scale_
        else:
            scaler.scale_ = scale_factor

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
    else:
        scale_factor = None

    return count_data, scale_factor


def standardize_data(
    count_data,
    target_sum=None,
    method='log',
    scale_factor=None
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
            scale=True,
            scale_factor=scale_factor
        )
    elif method == 'log_scale':
        return _normalize_for_pca(
            count_data,
            target_sum,
            log=True,
            scale=True,
            scale_factor=scale_factor
        )
    elif method == 'depth':
        return _normalize_for_pca(
            count_data,
            target_sum
        )
    elif method is None:
        return count_data, None
    else:
        raise ValueError(
            'method must be None, `depth`, `log`, `scale`, or `log_scale`, '
            f'{method} provided'
        )
