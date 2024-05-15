import numpy as np
import scipy.sparse as sps
import sklearn.metrics
import numpy.testing as npt
import anndata as ad
import unittest

from scself._noise2self.graph import local_optimal_knn
from scself._noise2self.common import (
    _dist_to_row_stochastic,
    _connect_to_row_stochastic,
    _invert_distance_graph,
    _search_k,
    standardize_data
)
from scself._noise2self import noise2self

M, N = 100, 10

RNG = np.random.default_rng(100)

BASE = RNG.negative_binomial(
    np.linspace(5, 50, N).astype(int),
    0.25,
    (M, N)

)

NOISE = RNG.negative_binomial(
    20,
    0.75,
    (M, N)
)

EXPR = BASE + NOISE
PEAKS = RNG.choice([0, 1], (M, N), p=(0.9, 0.1))

DIST = sklearn.metrics.pairwise_distances(EXPR, metric='cosine')
PDIST = sklearn.metrics.pairwise_distances(PEAKS, metric='cosine')

ADATA = ad.AnnData(EXPR.astype(int))


def _knn(k, dist=sps.csr_matrix(DIST)):
    return local_optimal_knn(
        dist.copy(),
        np.array([k] * 100),
        keep='smallest'
    )


class TestDistInvert(unittest.TestCase):

    def test_invert_sparse(self):
        graph = sps.csr_matrix(DIST)
        graph = _invert_distance_graph(graph).A

        invert_order = np.zeros_like(DIST)
        np.divide(1, DIST, out=invert_order, where=DIST != 0)

        for i in range(graph.shape[0]):
            npt.assert_equal(
                np.argsort(invert_order[i]),
                np.argsort(graph[i])
            )

    def test_invert_dense(self):
        graph = _invert_distance_graph(DIST.copy())

        invert_order = np.zeros_like(DIST)
        np.divide(1, DIST, out=invert_order, where=DIST != 0)

        for i in range(graph.shape[0]):
            npt.assert_equal(
                np.argsort(invert_order[i]),
                np.argsort(graph[i])
            )


class TestRowStochastic(unittest.TestCase):

    loss = 'mse'

    def test_full_k(self):
        graph = sps.csr_matrix(DIST)

        row_stochastic = _dist_to_row_stochastic(graph)
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.ones_like(row_sums), row_sums)
        self.assertEqual(len(row_sums), M)

        self.assertTrue(sps.isspmatrix_csr(row_stochastic))

    def test_full_k_connect(self):
        graph = sps.csr_matrix(DIST)

        row_stochastic = _connect_to_row_stochastic(graph)
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.ones_like(row_sums), row_sums)
        self.assertEqual(len(row_sums), M)

        self.assertTrue(sps.isspmatrix_csr(row_stochastic))

    def test_small_k(self):
        graph = _knn(3)

        npt.assert_array_equal(graph.getnnz(axis=1), np.full(M, 3))

        row_stochastic = _dist_to_row_stochastic(graph)
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.ones_like(row_sums), row_sums)
        self.assertEqual(len(row_sums), M)

        self.assertTrue(sps.isspmatrix_csr(row_stochastic))

    def test_small_k_connect(self):
        graph = _knn(3)

        npt.assert_array_equal(graph.getnnz(axis=1), np.full(M, 3))

        row_stochastic = _connect_to_row_stochastic(graph)
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.ones_like(row_sums), row_sums)
        self.assertEqual(len(row_sums), M)

        self.assertTrue(sps.isspmatrix_csr(row_stochastic))

    def test_zero_k(self):

        graph = _knn(3, dist=sps.csr_matrix((M, M), dtype=float))
        npt.assert_array_equal(graph.getnnz(), 0)

        row_stochastic = _dist_to_row_stochastic(graph)
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.zeros_like(row_sums), row_sums)

        self.assertTrue(sps.isspmatrix_csr(row_stochastic))

    def test_zero_k_connect(self):

        graph = _knn(3, dist=sps.csr_matrix((M, M), dtype=float))
        npt.assert_array_equal(graph.getnnz(), 0)

        row_stochastic = _connect_to_row_stochastic(graph)
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.zeros_like(row_sums), row_sums)

        self.assertTrue(sps.isspmatrix_csr(row_stochastic))


class _N2SSetup:

    data = EXPR.astype(float)
    dist = (DIST.copy(), )
    normalize = 'log'
    loss = 'mse'
    correct_loss = np.array([
        234.314,
        223.1832274,
        212.9209424,
        203.5771366,
        194.4068273,
        185.2830849
    ])
    correct_mse_argmin = 5
    correct_opt_pc = 7
    correct_opt_k = 4


class TestKNNSearch(_N2SSetup, unittest.TestCase):

    def test_ksearch_regression(self):

        mse = _search_k(
            self.data,
            self.dist,
            np.arange(1, 7),
            loss=self.loss
        )

        self.assertEqual(np.argmin(mse), self.correct_mse_argmin)

        npt.assert_almost_equal(
            self.correct_loss,
            mse
        )

    def test_ksearch_regression_sparse(self):

        mse = _search_k(
            sps.csr_matrix(self.data),
            self.dist,
            np.arange(1, 7),
            loss=self.loss
        )

        self.assertEqual(np.argmin(mse), self.correct_mse_argmin)

        npt.assert_almost_equal(
            self.correct_loss,
            mse
        )


class TestNoise2Self(_N2SSetup, unittest.TestCase):

    def test_knn_select_stack_regression(self):

        _, opt_pc, opt_k, local_ks = noise2self(
            self.data,
            np.arange(1, 11),
            np.array([3, 5, 7]),
            loss=self.loss,
            standardization_method=self.normalize
        )

        self.assertEqual(opt_pc, self.correct_opt_pc)
        self.assertEqual(opt_k, self.correct_opt_k)

    def test_knn_select_stack_regression_sparse(self):

        obsp, opt_pc, opt_k, local_ks = noise2self(
            sps.csr_matrix(self.data),
            np.arange(1, 11),
            np.array([3, 5, 7]),
            loss=self.loss,
            standardization_method=self.normalize
        )

        self.assertEqual(opt_pc, self.correct_opt_pc)
        self.assertEqual(opt_k, self.correct_opt_k)

    def test_knn_select_stack_regression_nopcsearch(self):

        _, opt_pc, opt_k, local_ks = noise2self(
            self.data,
            np.arange(1, 11),
            5,
            loss=self.loss,
            standardization_method=self.normalize
        )

        self.assertEqual(opt_pc, 5)
        self.assertIsNone(opt_k)


class TestKNNSearchNoNorm(TestNoise2Self):

    normalize = None
    data = standardize_data(
        ad.AnnData(EXPR.astype(np.float32))
    ).X


class TestKNNSearchLogLoss(TestKNNSearch, TestNoise2Self):

    normalize = None
    data = PEAKS.astype(float)
    dist = (PDIST.copy(), )
    loss = 'log_loss'
    correct_loss = np.array([
        0.999322,
        0.5909092,
        0.3082457,
        0.2716103,
        0.2355531,
        0.1897085
    ])
    correct_opt_pc = 7
    correct_opt_k = 8


class TestKNNSearchMultimodal(TestKNNSearch):

    dist = (DIST.copy(), DIST.copy())


class TestKNNSearchMultimodalRescale(TestKNNSearch):

    dist = (DIST.copy() * 10, DIST.copy() / 2)


class TestKNNSearchMultimodalEdge(TestKNNSearch):

    dist = (DIST.copy() * 10, sps.csr_array(DIST.shape))
