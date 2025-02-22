import unittest

import numpy as np
import scipy.sparse as sps

from scself.utils.correlation import cov

X = np.random.default_rng(100).random((500, 100))


class TestCOV(unittest.TestCase):

    def test_dense(self):

        x_cov = cov(X)
        np.testing.assert_almost_equal(
            x_cov,
            np.cov(X.T),
            decimal=4
        )

    def test_dense_axis1(self):

        x_cov = cov(X, axis=1)
        np.testing.assert_almost_equal(
            x_cov,
            np.cov(X),
            decimal=4
        )

    def test_sparse(self):

        x_cov = cov(sps.csr_array(X))
        np.testing.assert_almost_equal(
            x_cov,
            np.cov(X.T),
            decimal=4
        )


    def test_sparse_axis1(self):

        x_cov = cov(sps.csr_array(X), axis=1)
        np.testing.assert_almost_equal(
            x_cov,
            np.cov(X),
            decimal=4
        )