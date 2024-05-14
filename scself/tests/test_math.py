import unittest
import contextlib

import numpy as np
import numpy.testing as npt
import scipy.sparse as sps

from scself.utils import (
    variance,
    coefficient_of_variation,
    pairwise_metric,
    mcv_mean_error
)

N = 1000

DATA_SEED = np.random.default_rng(1010101).random(N)
DATA_SEED[0] = 0
DATA_SEED[-1] = 1

DATA_SPACE = [0.5, -1, 10, 100]

DATA = np.vstack((DATA_SEED * DATA_SPACE[0],
                  DATA_SEED * DATA_SPACE[1],
                  DATA_SEED * DATA_SPACE[2],
                  DATA_SEED * DATA_SPACE[3])).T


class TestMSENumba(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        rng = np.random.default_rng(12345)

        cls.X = rng.random((100, 20))
        cls.PC = rng.random((100, 10))
        cls.ROTATION = rng.random((10, 20))
        cls.Y = cls.PC @ cls.ROTATION
        cls.Z = np.mean((cls.X - cls.Y) ** 2)
        cls.MAE = np.mean(np.abs(cls.X - cls.Y))
        cls.Z_row = np.mean((cls.X - cls.Y) ** 2, axis=1)
        cls.MAE_row = np.mean(np.abs(cls.X - cls.Y), axis=1)
        cls.Z_noY = np.mean(cls.X ** 2)
        cls.Z_noY_row = np.mean(cls.X ** 2, axis=1)

    def test_mse_rowwise(self):

        npt.assert_array_almost_equal(
            mcv_mean_error(
                self.X,
                self.PC,
                self.ROTATION,
                axis=None,
                squared=True
            ),
            self.Z
        )

        npt.assert_array_almost_equal(
            mcv_mean_error(
                sps.csr_array(self.X),
                self.PC,
                self.ROTATION,
                by_row=True,
                squared=True
            ),
            self.Z_row
        )

        npt.assert_array_almost_equal(
            mcv_mean_error(
                sps.csr_array(self.X),
                self.PC,
                self.ROTATION,
                axis=None,
                squared=True
            ),
            self.Z
        )

    def test_mae_rowwise(self):

        npt.assert_array_almost_equal(
            mcv_mean_error(
                self.X,
                self.PC,
                self.ROTATION,
                axis=None,
                squared=False
            ),
            self.MAE
        )

        npt.assert_array_almost_equal(
            mcv_mean_error(
                sps.csr_array(self.X),
                self.PC,
                self.ROTATION,
                by_row=True,
                squared=False
            ),
            self.MAE_row
        )

        npt.assert_array_almost_equal(
            mcv_mean_error(
                sps.csr_array(self.X),
                self.PC,
                self.ROTATION,
                axis=None,
                squared=False
            ),
            self.MAE
        )


class TestMSE(unittest.TestCase):

    metric = 'mse'
    none_context = contextlib.nullcontext()

    @classmethod
    def setUpClass(cls) -> None:

        rng = np.random.default_rng(12345)

        cls.X = rng.random((100, 20))
        cls.Y = rng.random((100, 20))
        cls.Z = np.mean((cls.X - cls.Y) ** 2)
        cls.Z_row = np.mean((cls.X - cls.Y) ** 2, axis=1)
        cls.Z_noY = np.mean(cls.X ** 2)
        cls.Z_noY_row = np.mean(cls.X ** 2, axis=1)

    def test_dense_dense(self):

        npt.assert_array_almost_equal(
            pairwise_metric(self.X, self.Y, axis=None, metric=self.metric),
            self.Z
        )

        npt.assert_array_almost_equal(
            pairwise_metric(self.X, self.Y, metric=self.metric),
            self.Z_row
        )

        with self.none_context:
            npt.assert_array_almost_equal(
                pairwise_metric(self.X, None, axis=None, metric=self.metric),
                self.Z_noY
            )

            npt.assert_array_almost_equal(
                pairwise_metric(self.X, None, metric=self.metric),
                self.Z_noY_row
            )

    def test_sparse_dense(self):

        X = sps.csr_matrix(self.X)

        npt.assert_array_almost_equal(
            pairwise_metric(X, self.Y, axis=None, metric=self.metric),
            self.Z
        )

        npt.assert_array_almost_equal(
            pairwise_metric(X, self.Y, metric=self.metric),
            self.Z_row
        )

        with self.none_context:
            npt.assert_array_almost_equal(
                pairwise_metric(X, None, axis=None, metric=self.metric),
                self.Z_noY
            )

            npt.assert_array_almost_equal(
                pairwise_metric(X, None, metric=self.metric),
                self.Z_noY_row
            )

    def test_sparse_sparse(self):

        X = sps.csr_matrix(self.X)
        Y = sps.csr_matrix(self.Y)

        npt.assert_array_almost_equal(
            pairwise_metric(X, Y, axis=None, metric=self.metric),
            self.Z
        )

        npt.assert_array_almost_equal(
            pairwise_metric(X, Y, metric=self.metric),
            self.Z_row
        )


class TestMAE(TestMSE):

    metric = 'mae'

    @classmethod
    def setUpClass(cls) -> None:

        rng = np.random.default_rng(12345)

        cls.X = rng.random((100, 20))
        cls.Y = rng.random((100, 20))

        cls.Z = np.mean(np.abs(cls.X - cls.Y))
        cls.Z_row = np.mean(np.abs(cls.X - cls.Y), axis=1)
        cls.Z_noY = np.mean(np.abs(cls.X))
        cls.Z_noY_row = np.mean(np.abs(cls.X), axis=1)


class TestLogLoss(TestMSE):

    metric = 'log_loss'

    @classmethod
    def setUpClass(cls) -> None:

        rng = np.random.default_rng(12345)

        cls.X = rng.choice([0, 1], (100, 20))
        cls.Y = rng.random((100, 20))

        cls.Z = cls.X * np.log(cls.Y)
        cls.Z += (1 - cls.X) * np.log(1 - cls.Y)
        cls.Z *= -1

        cls.Z_row = np.mean(cls.Z, axis=1)
        cls.Z = np.mean(cls.Z_row)

        cls.Z_noY = None
        cls.Z_noY_row = None

    def setUp(self) -> None:
        self.none_context = self.assertRaises(ValueError)
        return super().setUp()


class TestVariance(unittest.TestCase):

    def setUp(self) -> None:

        self.arr = np.random.rand(50, 5)
        self.arr[self.arr < 0.5] = 0

        self.sarr = sps.csr_matrix(self.arr)
        return super().setUp()

    def test_flattened(self):

        npt.assert_almost_equal(
            variance(self.arr),
            variance(self.sarr)
        )

    def test_axis_0(self):

        npt.assert_almost_equal(
            variance(self.arr, axis=0),
            variance(self.sarr, axis=0)
        )

    def test_axis_1(self):

        npt.assert_almost_equal(
            variance(self.arr, axis=1),
            variance(self.sarr, axis=1)
        )


class TestCV(unittest.TestCase):

    def setUp(self) -> None:

        self.arr = np.random.rand(50, 5)
        self.arr[self.arr < 0.5] = 0

        self.sarr = sps.csr_matrix(self.arr)
        return super().setUp()

    def test_flattened(self):

        npt.assert_almost_equal(
            coefficient_of_variation(self.arr),
            coefficient_of_variation(self.sarr)
        )

    def test_axis_0(self):

        npt.assert_almost_equal(
            coefficient_of_variation(self.arr, axis=0),
            coefficient_of_variation(self.sarr, axis=0)
        )

    def test_axis_1(self):

        npt.assert_almost_equal(
            coefficient_of_variation(self.arr, axis=1),
            coefficient_of_variation(self.sarr, axis=1)
        )
