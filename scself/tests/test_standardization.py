import unittest
import numpy as np
import numpy.testing as npt
import anndata as ad
import scipy.sparse as sps

from scself.utils import standardize_data
from scself import TruncRobustScaler

X = np.random.default_rng(100).integers(0, 5, (100, 20))

COUNT = X.sum(1)

TS = np.full(100, 50)
SF = COUNT / TS
SCALE = TruncRobustScaler(with_centering=False).fit(
    np.divide(X, SF[:, None])
).scale_
LOG_SCALE = TruncRobustScaler(with_centering=False).fit(
    np.log1p(np.divide(X, SF[:, None]))
).scale_


def _equal(x, y):

    if sps.issparse(x):
        x = x.A
    if sps.issparse(y):
        y = y.A

    npt.assert_array_almost_equal(
        x,
        y
    )


class TestScalingDense(unittest.TestCase):

    def setUp(self) -> None:
        self.data = ad.AnnData(X.copy())
        self.data.layers['a'] = X.copy()
        return super().setUp()

    def test_depth(self):

        standardize_data(self.data, target_sum=50, method='depth')
        _equal(
            np.divide(X, SF[:, None]),
            self.data.X
        )
        _equal(
            SF,
            self.data.obs['X_size_factor'].values
        )

    def test_depth_with_size_factor(self):

        standardize_data(
            self.data,
            size_factor=np.ones_like(SF),
            method='depth'
        )
        _equal(
            X,
            self.data.X
        )
        _equal(
            np.ones_like(SF),
            self.data.obs['X_size_factor'].values
        )

    def test_log1p(self):

        standardize_data(self.data, target_sum=50, method='log')
        _equal(
            np.log1p(np.divide(X, SF[:, None])),
            self.data.X
        )

    def test_scale(self):

        standardize_data(self.data, target_sum=50, method='scale')
        _equal(
            np.divide(np.divide(X, SF[:, None]), SCALE[None, :]),
            self.data.X
        )
        _equal(
            SCALE,
            self.data.var['X_scale_factor'].values
        )

    def test_scale_with_factor(self):

        standardize_data(
            self.data,
            target_sum=50,
            method='scale',
            scale_factor=np.ones_like(SCALE)
        )
        _equal(
            np.divide(X, SF[:, None]),
            self.data.X
        )
        _equal(
            np.ones_like(SCALE),
            self.data.var['X_scale_factor'].values
        )

    def test_log_scale(self):

        standardize_data(self.data, target_sum=50, method='log_scale')
        _equal(
            np.divide(
                np.log1p(
                    np.divide(X, SF[:, None])
                ),
                LOG_SCALE[None, :]
            ),
            self.data.X
        )
        _equal(
            LOG_SCALE,
            self.data.var['X_scale_factor'].values
        )

    def test_none(self):

        standardize_data(self.data, target_sum=50, method=None)
        _equal(
            X,
            self.data.X
        )

    def test_layer(self):

        standardize_data(
            self.data,
            target_sum=50,
            method='log_scale',
            layer='a'
        )
        _equal(
            self.data.X,
            X
        )
        _equal(
            np.divide(
                np.log1p(
                    np.divide(X, SF[:, None])
                ),
                LOG_SCALE[None, :]
            ),
            self.data.layers['a']
        )
        _equal(
            LOG_SCALE,
            self.data.var['a_scale_factor'].values
        )

    def test_subset_depth(self):

        standardize_data(
            self.data,
            target_sum=20,
            method='log',
            subset_genes_for_depth=['0', '1', '2']
        )

        sf = X[:, 0:3].sum(1) / 20
        sf[sf == 0] = 1.

        _equal(
            np.log1p(
                np.divide(X, sf[:, None])
            ),
            self.data.X
        )
        _equal(
            sf,
            self.data.obs['X_size_factor'].values
        )
        _equal(
            COUNT,
            self.data.obs['X_counts'].values
        )
        _equal(
            X[:, 0:3].sum(1),
            self.data.obs['X_subset_counts'].values
        )


class TestScalingCSR(TestScalingDense):

    def setUp(self) -> None:
        self.data = ad.AnnData(sps.csr_matrix(X))
        self.data.layers['a'] = sps.csr_matrix(X)


class TestScalingCSC(TestScalingDense):

    def setUp(self) -> None:
        self.data = ad.AnnData(sps.csc_matrix(X))
        self.data.layers['a'] = sps.csc_matrix(X)
