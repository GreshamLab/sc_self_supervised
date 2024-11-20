import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    MinMaxScaler
)


class TruncRobustScaler(RobustScaler):

    def fit(self, X, y=None):

        if isinstance(X, (sps.csr_matrix, sps.csr_array)):
            # Use custom extractor to turn X into a CSC with no
            # indices array; RobustScaler makes an undesirabe
            # CSR->CSC conversion
            from ..sparse.math import sparse_csr_extract_columns
            super().fit(
                sparse_csr_extract_columns(X, fake_csc_matrix=True),
                y
            )
        else:
            super().fit(
                X,
                y
            )

        # Use StandardScaler to deal with sparse & dense
        # There are C extensions for CSR variance without copy
        _std_scale = StandardScaler(with_mean=False).fit(X)

        _post_robust_var = _std_scale.var_ / (self.scale_ ** 2)
        _rescale_idx = _post_robust_var > 1

        _scale_mod = np.ones_like(self.scale_)
        _scale_mod[_rescale_idx] = np.sqrt(_post_robust_var[_rescale_idx])

        self.scale_ *= _scale_mod

        return self


class TruncStandardScaler(StandardScaler):

    def fit(self, X, y=None):

        super().fit(
            X,
            y
        )

        self.scale_[self.var_ <= 1] = 1

        return self


class TruncMinMaxScaler(MinMaxScaler):

    def __init__(
        self,
        feature_range=(0, 1),
        quantile_range=(0.01, 0.99),
        clipping_range=None,
        *,
        copy=True,
        clip=False
    ):
        if (
            quantile_range is not None and
            clipping_range is not None
        ):
            raise ValueError(
                "Provide quantile_range or clipping_range, not both"
            )

        self.feature_range = feature_range
        self.quantile_range = quantile_range
        self.clipping_range = clipping_range
        self.copy = copy
        self.clip = True

    def fit(self, X, y=None):


        # If there's no truncation ranges
        # just run MinMaxScaler
        if (
            self.quantile_range is None and
            self.clipping_range is None
        ):
            return super().fit(X, y)

        # Get the data_min and data_max for
        # truncation ranges given as quantiles
        elif self.quantile_range is not None:

            _bottom_quantile, _top_quantile = self.quantile_range

            if _bottom_quantile is None or (_bottom_quantile == 0.):
                data_min_ = np.nanmin(X, axis=0)
            else:
                data_min_ = np.nanquantile(
                    X, _bottom_quantile, axis=0, method='lower'
                )
    
            if _top_quantile is None or (_bottom_quantile == 1.):
                data_min_ = np.nanmax(X, axis=0)
            else:
                data_max_ = np.nanquantile(
                    X, _top_quantile, axis=0, method='higher'
                )
        
        # Set data_min and data_max if truncation range
        # is given as raw values
        elif self.clipping_range is not None:
            n = X.shape[1]
            print(_to_array(self.clipping_range, n))
            data_min_, data_max_ = _to_array(self.clipping_range, n)

        else:
            raise ValueError(
                "quantile_range and clipping_range cannot both be None"
            )

        data_range_ = data_max_ - data_min_

        _fixed_range = data_range_.copy()
        _fixed_range[_fixed_range == 0] = 1.

        self.scale_ = self.feature_range[1] - self.feature_range[0]
        self.scale_ /= _fixed_range
        self.min_ = self.feature_range[0] - data_min_ * self.scale_

        self.data_max_ = data_max_
        self.data_min_ = data_min_
        self.data_range_ = data_range_

        return self


def _to_array(x, n):

    if isinstance(x, (tuple, list)):
        return tuple(_to_array(y, n) for y in x)
    elif not isinstance(x, np.ndarray):
        return np.repeat(x, n)
    else:
        return x
