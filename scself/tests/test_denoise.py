import unittest
import numpy as np
import sklearn.metrics
import scipy.sparse as sps
import numpy.testing as npt

from scself._noise2self.graph import local_optimal_knn
from scself import denoise_data

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

DIST = sklearn.metrics.pairwise_distances(EXPR, metric='cosine')


def _knn(k, dist=sps.csr_matrix(DIST)):
    return local_optimal_knn(
        dist.copy(),
        np.array([k] * 100),
        keep='smallest'
    )


class TestDenoiser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        graph = _knn(3).A
        graph = np.divide(1, graph, out=graph, where=graph != 0)
        graph /= graph.sum(1)[:, None]

        cls.denoised = graph @ EXPR

        cls.denoised_zeros = cls.denoised.copy()
        cls.denoised_zeros[cls.denoised_zeros < 50] = 0

    def test_denoise_eye(self):

        denoised = denoise_data(
            EXPR.astype(np.float64),
            np.eye(M)
        )

        npt.assert_equal(denoised, EXPR.astype(np.float64))

    def test_denoise_blank(self):

        denoised = denoise_data(
            EXPR.astype(np.float64),
            np.zeros((M, M), float)
        )

        npt.assert_equal(denoised, np.zeros_like(denoised))

    def test_denoise_knn(self):

        denoised = denoise_data(
            EXPR.astype(np.float64),
            _knn(3)
        )

        npt.assert_almost_equal(denoised, self.denoised)

    def test_denoise_knn_0s(self):

        denoised = denoise_data(
            EXPR.astype(np.float64),
            _knn(3),
            zero_threshold=50
        )

        npt.assert_almost_equal(denoised, self.denoised_zeros)

    def test_denoise_knn_chunky(self):

        denoised = denoise_data(
            EXPR.astype(np.float64),
            _knn(3),
            chunk_size=5
        )

        npt.assert_almost_equal(denoised, self.denoised)

    def test_denoise_knn_chunky_0s(self):

        denoised = denoise_data(
            EXPR.astype(np.float64),
            _knn(3),
            zero_threshold=50,
            chunk_size=5
        )

        npt.assert_almost_equal(denoised, self.denoised_zeros)

    def test_denoise_dual_knn(self):

        denoised = denoise_data(
            [EXPR.astype(np.float64), EXPR.astype(np.float64)],
            [_knn(3), _knn(3)]
        )

        npt.assert_almost_equal(denoised[0], self.denoised)
        npt.assert_almost_equal(denoised[1], self.denoised)

    def test_denoise_dual_knn_chunky(self):

        denoised = denoise_data(
            [EXPR.astype(np.float64), EXPR.astype(np.float64)],
            [_knn(3), _knn(3)],
            chunk_size=5
        )

        npt.assert_almost_equal(denoised[0], self.denoised)
        npt.assert_almost_equal(denoised[1], self.denoised)

    def test_denoise_dual_knn_0s(self):

        denoised = denoise_data(
            [EXPR.astype(np.float64), EXPR.astype(np.float64)],
            [_knn(3), _knn(3)],
            zero_threshold=50
        )

        npt.assert_almost_equal(denoised[0], self.denoised_zeros)
        npt.assert_almost_equal(denoised[1], self.denoised_zeros)

    def test_denoise_dual_knn_chunky_0s(self):

        denoised = denoise_data(
            [EXPR.astype(np.float64), EXPR.astype(np.float64)],
            [_knn(3), _knn(3)],
            zero_threshold=50,
            chunk_size=5
        )

        npt.assert_almost_equal(denoised[0], self.denoised_zeros)
        npt.assert_almost_equal(denoised[1], self.denoised_zeros)
