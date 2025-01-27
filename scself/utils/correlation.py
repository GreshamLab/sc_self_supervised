import numpy as np

### COV AND CORRCOEF ###
### LIGHER WEIGHT THAN NUMPY ###

def cov(X, axis=0):

    # Center and get num rows
    avg, w_sum = np.average(X, axis=axis, weights=None, returned=True)
    w_sum = w_sum[0]
    X = X - (avg[None, :] if axis == 0 else avg[:, None])

    # Gram matrix
    X = np.dot(X.T, X)
    X *= np.true_divide(1, w_sum)

    return X

def corrcoef(X, axis=0):

    X = cov(X, axis=axis)
    sd = np.sqrt(np.diag(X))

    _zero_var = sd == 0
    sd[_zero_var] = 1.

    X /= sd[:, None]
    X /= sd[None, :]

    # Fixes for float precision
    np.clip(X, -1, 1, out=X)
    np.fill_diagonal(X, 1.)

    return X
