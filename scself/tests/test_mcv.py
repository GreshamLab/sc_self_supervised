import unittest
import numpy as np
import numpy.testing as npt
import scipy.sparse as sps
import anndata as ad

from scself import TruncRobustScaler, mcv
from scself._mcv.common import mcv_comp
from scself.utils import mcv_mean_error
from scself.utils.standardization import _normalize_for_pca

from ._stubs import (
    COUNTS
)


def _safe_sum(x, axis):

    sums = x.sum(axis)

    try:
        sums = sums.A1
    except AttributeError:
        pass

    return sums


def _safe_dense(x):

    try:
        return x.A
    except AttributeError:
        return x


class TestMCV(unittest.TestCase):

    def test_sparse_log(self):
        data = sps.csr_matrix(COUNTS)
        self.assertEqual(
            np.argmin(
                mcv(data, n=1, n_pcs=5, silent=True)
            ),
            1
        )

    def test_sparse_log_scale(self):
        data = sps.csr_matrix(COUNTS)

        self.assertEqual(
            np.argmin(
                mcv(
                    data,
                    n=1,
                    n_pcs=5,
                    standardization_method='log_scale',
                    silent=True
                )
            ),
            1
        )


class TestMCVMetrics(unittest.TestCase):

    def setUp(self):
        self.data = sps.csr_matrix(COUNTS)

    def testMSErow(self):

        mse = mcv_mean_error(
            self.data,
            self.data @ np.zeros((10, 10)),
            np.eye(10),
            squared=True
        )

        npt.assert_almost_equal(
            self.data.power(2).sum(axis=1).A1 / 10,
            mse
        )

    def testMSER2row(self):

        mse, r2 = mcv_comp(
            self.data,
            self.data @ np.zeros((10, 10)),
            np.eye(10),
            'mse',
            axis=1,
            calculate_r2=True
        )

        calc_mse = self.data.power(2).sum(axis=1).A1

        npt.assert_almost_equal(
            calc_mse / 10,
            mse
        )

        calc_r2 = calc_mse / self.data.power(2).sum(axis=1).A1
        calc_r2 = 1 - calc_r2

        npt.assert_almost_equal(
            r2,
            calc_r2
        )


class TestMCVStandardization(unittest.TestCase):

    tol = 6

    def setUp(self):
        super().setUp()
        self.data = ad.AnnData(sps.csr_matrix(COUNTS))

    def test_depth(self):

        _normalize_for_pca(self.data, target_sum=100, log=False, scale=False)

        rowsums = _safe_sum(self.data.X, 1)

        npt.assert_almost_equal(
            rowsums,
            np.full_like(rowsums, 100.),
            decimal=self.tol
        )

    def test_depth_log(self):

        _normalize_for_pca(self.data, target_sum=100, log=True, scale=False)

        rowsums = _safe_sum(self.data.X, 1)

        npt.assert_almost_equal(
            rowsums,
            np.log1p(100 * COUNTS / np.sum(COUNTS, axis=1)[:, None]).sum(1),
            decimal=self.tol
        )

    def test_depth_scale(self):

        _normalize_for_pca(self.data, target_sum=100, log=False, scale=True)

        npt.assert_almost_equal(
            _safe_dense(self.data.X),
            TruncRobustScaler(with_centering=False).fit_transform(
                100 * COUNTS / np.sum(COUNTS, axis=1)[:, None]
            ),
            decimal=self.tol
        )

    def test_depth_log_scale(self):

        _normalize_for_pca(self.data, target_sum=100, log=True, scale=True)

        npt.assert_almost_equal(
            _safe_dense(self.data.X),
            TruncRobustScaler(with_centering=False).fit_transform(
                np.log1p(100 * COUNTS / np.sum(COUNTS, axis=1)[:, None])
            ),
            decimal=self.tol
        )


class TestMCVStandardizationDense(TestMCVStandardization):

    tol = 4

    def setUp(self):
        super().setUp()
        self.data = ad.AnnData(COUNTS.copy())


class TestMCVStandardizationCSC(TestMCVStandardization):

    tol = 4

    def setUp(self):
        super().setUp()
        self.data = ad.AnnData(sps.csc_matrix(COUNTS))
