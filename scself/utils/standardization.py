import warnings
import numpy as np
import scipy.sparse as sps

from scself.sparse import is_csr
from scself.scaling import TruncRobustScaler
from scself.utils.sum import array_sum
from scself.utils.cast_dtype_inplace import cast_to_float_inplace


def _normalize(
    count_data,
    target_sum=None,
    log=False,
    scale=False,
    scale_factor=None,
    size_factor=None,
    subset_genes_for_depth=None,
    layer='X'
):
    """
    Depth normalize and log pseudocount
    This operation will be entirely inplace

    :param count_data: Integer data
    :type count_data: ad.AnnData
    :return: Standardized data
    :rtype: np.ad.AnnData
    """

    if target_sum is not None and size_factor is not None:
        warnings.warn(
            "target_sum has no effect when size_factor is passed"
        )

    lref = _get_layer(count_data, layer)

    if subset_genes_for_depth is not None and size_factor is None:
        sub_counts, size_factor = _size_factors(
            _get_layer(count_data[:, subset_genes_for_depth], layer),
            target_sum=target_sum
        )
        counts = array_sum(lref, 1)
        count_data.obs[f'{layer}_subset_counts'] = sub_counts
    elif size_factor is None:
        counts, size_factor = _size_factors(
            lref,
            target_sum=target_sum
        )
    else:
        counts = array_sum(lref, 1)

    count_data.obs[f'{layer}_counts'] = counts
    count_data.obs[f'{layer}_size_factor'] = size_factor

    if is_csr(lref):
        from ..sparse.math import sparse_normalize_total
        sparse_normalize_total(
            lref,
            size_factor=size_factor
        )

    elif layer == 'X':
        count_data.X = _normalize_total(
            lref,
            size_factor=size_factor
        )

    else:
        count_data.layers[layer] = _normalize_total(
            lref,
            size_factor=size_factor
        )

    if log:
        log1p(
            _get_layer(count_data, layer)
        )

    if scale:
        lref = _get_layer(count_data, layer)

        scaler = TruncRobustScaler(with_centering=False)

        if scale_factor is None:
            scaler.fit(lref)
            scale_factor = scaler.scale_
        else:
            scaler.scale_ = scale_factor

        if is_csr(lref):
            from ..sparse.math import sparse_normalize_columns
            sparse_normalize_columns(
                lref,
                scaler.scale_
            )
        elif layer == 'X':
            count_data.X = scaler.transform(
                lref
            )
        else:
            count_data.layers[layer] = scaler.transform(
                lref
            )

        count_data.var[f'{layer}_scale_factor'] = scaler.scale_
    else:
        scale_factor = None

    return count_data, scale_factor


def standardize_data(
    count_data,
    target_sum=None,
    method='log',
    scale_factor=None,
    size_factor=None,
    subset_genes_for_depth=None,
    layer='X'
):
    """
    Standardize single cell data. Note that this is in-place and
    overwrites existing data object.

    :param count_data: Integer data in an AnnData object
    :type count_data: ad.AnnData
    :param target_sum: Target sum for depth normalization,
        None uses median counts, defaults to None
    :type target_sum: int, optional
    :param method: Standardization method, 'log', 'scale',
        'log_scale', and 'depth' available. None returns data
        unmodified. Defaults to 'log'
    :type method: str, optional
    :param scale_factor: Gene scale factors for 'scale' methods,
        None fits scale factors with TruncRobustScaler. defaults to None
    :type scale_factor: np.ndarray, optional
    :param size_factor: Cell size factors. If passed, target_sum
        has no effect. Defaults to None
    :type size_factor: np.ndarray, optional
    :param subset_genes_for_depth: _description_, defaults to None
    :type subset_genes_for_depth: _type_, optional
    :param layer: _description_, defaults to 'X'
    :type layer: str, optional
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """

    if method == 'log':
        return _normalize(
            count_data,
            target_sum=target_sum,
            log=True,
            size_factor=size_factor,
            subset_genes_for_depth=subset_genes_for_depth,
            layer=layer
        )
    elif method == 'scale':
        return _normalize(
            count_data,
            target_sum=target_sum,
            scale=True,
            scale_factor=scale_factor,
            size_factor=size_factor,
            subset_genes_for_depth=subset_genes_for_depth,
            layer=layer
        )
    elif method == 'log_scale':
        return _normalize(
            count_data,
            target_sum=target_sum,
            log=True,
            scale=True,
            scale_factor=scale_factor,
            size_factor=size_factor,
            subset_genes_for_depth=subset_genes_for_depth,
            layer=layer
        )
    elif method == 'depth':
        return _normalize(
            count_data,
            target_sum=target_sum,
            size_factor=size_factor,
            subset_genes_for_depth=subset_genes_for_depth,
            layer=layer
        )
    elif method is None:
        return count_data, None
    else:
        raise ValueError(
            'method must be None, `depth`, `log`, `scale`, or `log_scale`, '
            f'{method} provided'
        )


def _normalize_total(
    data,
    target_sum=None,
    size_factor=None
):

    if size_factor is None:
        _, size_factor = _size_factors(data, target_sum=target_sum)

    if sps.issparse(data):
        return data.multiply((1/size_factor)[:, None])
    else:
        cast_to_float_inplace(data)
        return np.divide(data, size_factor[:, None], out=data)


def _size_factors(data, target_sum=None):
    counts = array_sum(data, 1)

    if target_sum is None:
        target_sum = np.median(counts)

    size_factor = counts / target_sum
    size_factor[counts == 0] = 1.

    return counts, size_factor


def log1p(data):

    if sps.issparse(data):
        data = data.data

    np.log1p(data, out=data)

    return data


def _get_layer(adata, layer):
    if layer == 'X':
        return adata.X
    else:
        return adata.layers[layer]
