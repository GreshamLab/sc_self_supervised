import numpy as np
import scipy.sparse as sps


def _log_loss(x, y, axis=1):

    if y is None:
        raise ValueError(
            "Cannot calculate log loss for only labels"
        )

    try:
        x = x.A
    except AttributeError:
        pass

    try:
        y = y.A
    except AttributeError:
        pass

    y = np.minimum(y, 1 - 1e-7)
    y = np.maximum(y, 1e-7)

    err = np.multiply(
        1 - x,
        np.log(1 - y)
    )
    err += np.multiply(
        x,
        np.log(y)
    )
    err = err.sum(axis=axis)
    err *= -1
    return err


def _mse(x, y, axis=1):

    if y is not None:
        ssr = x - y
    else:
        ssr = x.copy()

    if sps.issparse(ssr):
        ssr.data **= 2
    elif isinstance(ssr, np.matrix):
        ssr = ssr.A
        ssr **= 2
    else:
        ssr **= 2

    return ssr.sum(axis=axis)


def _mae(x, y, axis=1):

    if y is not None:
        ssr = x - y
    else:
        ssr = x

    ssr = np.abs(ssr)

    return ssr.sum(axis=axis)


def pairwise_metric(
    x,
    y,
    metric='mse',
    axis=1,
    **kwargs
):
    """
    Pairwise metric between two arrays

    :param x: _description_
    :type x: _type_
    :param y: _description_
    :type y: _type_
    :param metric: _description_, defaults to 'mse'
    :type metric: str, optional
    :return: _description_
    :rtype: _type_
    """

    if metric == 'mse':
        metric = _mse
    elif metric == 'mae':
        metric = _mae
    elif metric == 'log_loss':
        metric = _log_loss

    loss = metric(
        x,
        y,
        axis=axis,
        **kwargs
    )

    try:
        loss = loss.A1
    except AttributeError:
        pass

    if axis is not None:
        return loss / x.shape[axis]
    else:
        _size = x.shape[0] * x.shape[1]
        return loss / _size


def variance(
    X,
    axis=None,
    ddof=0
):
    """
    Function to calculate variance for sparse or dense arrays

    :param X: Sparse or dense data array
    :type X: np.ndarray, sp.spmatrix
    :param axis: Across which axis (None flattens),
        defaults to None
    :type axis: int, optional
    :param ddof: Delta degrees of freedom,
        defaults to 0
    :type ddof: int, optional
    :return: Variance or vector of variances across an axis
    :rtype: numeric, np.ndarray
    """

    if sps.issparse(X):

        _n = np.prod(X.shape)

        if axis is None:
            _mean = X.mean()
            _nz = _n - X.size

            _var = np.sum(np.power(X.data - _mean, 2))
            _var += np.power(_mean, 2) * _nz

            return _var / (_n - ddof)

        else:
            _mean = X.mean(axis=axis).A1
            _nz = -1 * X.getnnz(axis=axis) + X.shape[axis]

            # Make a sparse mask of means over axis
            _mean_mask = sps.csr_matrix(
                ((np.ones(X.data.shape, dtype=float), X.indices, X.indptr))
            )

            if axis == 0:
                _mean_mask = _mean_mask.multiply(_mean[np.newaxis, :])
            else:
                _mean_mask = _mean_mask.multiply(_mean[:, np.newaxis])

            _var = (X - _mean_mask).power(2).sum(axis=axis).A1
            _var += np.power(_mean, 2) * _nz

        return _var / (X.shape[axis] - ddof)

    else:

        return np.var(X, axis=axis, ddof=ddof)


def coefficient_of_variation(
    X,
    axis=None,
    ddof=0
):
    """
    Calculate coefficient of variation

    :param X: Sparse or dense data array
    :type X: np.ndarray, sp.spmatrix
    :param axis: Across which axis (None flattens),
        defaults to None
    :type axis: int, optional
    :param ddof: Delta degrees of freedom,
        defaults to 0
    :type ddof: int, optional
    :return: CV or vector of CVs across an axis
    :rtype: numeric, np.ndarray
    """

    _var = variance(X, axis=axis, ddof=ddof)
    _mean = X.mean(axis=axis)

    try:
        _mean = _mean.A1
    except AttributeError:
        pass

    return np.divide(
        np.sqrt(_var),
        _mean,
        out=np.zeros_like(_var),
        where=_mean != 0
    )


def mcv_mean_error(
    x,
    pc,
    rotation,
    axis=1,
    squared=True,
    **metric_kwargs
):

    if sps.issparse(x):

        from ..sparse.math import mcv_mean_error_sparse

        return mcv_mean_error_sparse(
            x,
            pc,
            rotation,
            axis=axis,
            squared=squared,
            **metric_kwargs
        )

    else:

        return pairwise_metric(
            x,
            pc @ rotation,
            metric='mse' if squared else 'mae',
            axis=axis,
            **metric_kwargs
        )
