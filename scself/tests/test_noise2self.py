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
    _noise_to_self_error,
    standardize_data,
    row_normalize
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


def _knn(k, dist=sps.csr_matrix(DIST)):
    return local_optimal_knn(
        dist.copy(),
        np.array([k] * 100),
        keep='smallest'
    )


class TestKnnSelect(unittest.TestCase):

    def test_nnz(self):

        knn_1 = _knn(1)
        npt.assert_equal(knn_1.indptr, np.arange(101))

        knn_2 = _knn(2)
        npt.assert_equal(knn_2.indptr, np.arange(101) * 2)

        for i in range(100):
            npt.assert_equal(
                sorted(knn_2[i].indices),
                sorted(np.argsort(DIST[i, :])[1:3])
            )

        knn_7 = _knn(7)
        npt.assert_equal(knn_7.indptr, np.arange(101) * 7)


class TestError(unittest.TestCase):

    def test_error_max_dense(self):

        err = _noise_to_self_error(
            EXPR.astype(np.float64),
            np.zeros_like(DIST),
            by_row=True
        )

        expect = np.mean(EXPR ** 2, axis=1)

        npt.assert_almost_equal(err, expect)

    def test_error_max_sparse(self):

        err = _noise_to_self_error(
            sps.csr_array(EXPR.astype(np.float64)),
            sps.csr_array(np.zeros_like(DIST)),
            chunk_size=5,
            by_row=True
        )

        expect = np.mean(EXPR ** 2, axis=1)

        npt.assert_almost_equal(err, expect)

    def test_error_max_sparse_nochunk(self):

        err = _noise_to_self_error(
            sps.csr_array(EXPR.astype(np.float64)),
            sps.csr_array(np.zeros_like(DIST)),
            chunk_size=None,
            by_row=True
        )

        expect = np.mean(EXPR ** 2, axis=1)

        npt.assert_almost_equal(err, expect)

    def test_error_min_dense(self):

        err = _noise_to_self_error(
            EXPR.astype(np.float64),
            np.eye(DIST.shape[0]),
            by_row=True
        )

        expect = np.zeros_like(err)

        npt.assert_almost_equal(err, expect)

    def test_error_min_sparse(self):

        err = _noise_to_self_error(
            sps.csr_array(EXPR.astype(np.float64)),
            sps.csr_array(np.eye(DIST.shape[0])),
            chunk_size=5,
            by_row=True
        )

        expect = np.zeros_like(err)

        npt.assert_almost_equal(err, expect)

    def test_error_min_sparse_nochunk(self):

        err = _noise_to_self_error(
            sps.csr_array(EXPR.astype(np.float64)),
            sps.csr_array(np.eye(DIST.shape[0])),
            chunk_size=None,
            by_row=True
        )

        expect = np.zeros_like(err)

        npt.assert_almost_equal(err, expect)


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

    def test_row_normalize(self):
        graph = sps.csr_matrix(DIST)
        graph = row_normalize(graph).A

        invert_order = np.zeros_like(DIST)
        np.divide(1, DIST, out=invert_order, where=DIST != 0)

        for i in range(graph.shape[0]):
            npt.assert_equal(
                np.argsort(invert_order[i]),
                np.argsort(graph[i])
            )

        _rowsum = graph.sum(axis=1)
        npt.assert_almost_equal(_rowsum, np.ones_like(_rowsum))

    def test_row_normalize_dense(self):
        graph = row_normalize(DIST.copy())

        invert_order = np.zeros_like(DIST)
        np.divide(1, DIST, out=invert_order, where=DIST != 0)

        for i in range(graph.shape[0]):
            npt.assert_equal(
                np.argsort(invert_order[i]),
                np.argsort(graph[i])
            )

        _rowsum = graph.sum(axis=1)
        npt.assert_almost_equal(_rowsum, np.ones_like(_rowsum))


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
        160.764803,
        142.8430503,
        135.422603,
        130.4516997,
        130.762186
    ])
    correct_mse_argmin = 4
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

        (_, opt_pc, opt_k, local_ks), errs = noise2self(
            self.data,
            np.arange(4, 11),
            np.array([3, 5, 7]),
            loss=self.loss,
            standardization_method=self.normalize,
            return_errors=True,
            random_state=50
        )

        self.assertEqual(opt_pc, self.correct_opt_pc)
        self.assertEqual(opt_k, self.correct_opt_k)

    def test_knn_select_stack_regression_sparse(self):

        obsp, opt_pc, opt_k, local_ks = noise2self(
            sps.csr_matrix(self.data),
            np.arange(4, 11),
            np.array([3, 5, 7]),
            loss=self.loss,
            standardization_method=self.normalize
        )

        self.assertEqual(opt_pc, self.correct_opt_pc)
        self.assertEqual(opt_k, self.correct_opt_k)

    def test_knn_select_stack_regression_nopcsearch(self):

        _, opt_pc, opt_k, local_ks = noise2self(
            self.data,
            np.arange(4, 11),
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
        0.4702795,
        0.2203677,
        0.1766228,
        0.1764214,
        0.1564008
    ])
    correct_opt_pc = 7
    correct_opt_k = 8
    correct_mse_argmin = 5


class TestKNNSearchMultimodal(TestKNNSearch):

    dist = (DIST.copy(), DIST.copy())


class TestKNNSearchMultimodalRescale(TestKNNSearch):

    dist = (DIST.copy() * 10, DIST.copy() / 2)


class TestKNNSearchMultimodalEdge(TestKNNSearch):

    dist = (DIST.copy() * 10, sps.csr_array(DIST.shape))
