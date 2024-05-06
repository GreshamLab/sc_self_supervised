import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler
)


class TruncRobustScaler(RobustScaler):

    def fit(self, X, y=None):

        if isinstance(X, (sps.csr_matrix, sps.csc_array)):
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
