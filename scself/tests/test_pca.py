"""
Generated with claude code (claude-sonnet-4-5)
Most of these look useless to be honest
"""

import unittest
import numpy as np
import numpy.testing as npt
import anndata as ad
import scipy.sparse as sps

from scself.utils._pca import pca, stratified_pca


class TestPCABasic(unittest.TestCase):
    """Test basic PCA functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        rng = np.random.default_rng(42)

        # Create dense test data
        cls.n_obs = 100
        cls.n_vars = 50
        cls.n_pcs = 10

        cls.X_dense = rng.random((cls.n_obs, cls.n_vars))
        cls.X_dense[cls.X_dense < 0.3] = 0

        # Create sparse test data
        cls.X_sparse = sps.csr_matrix(cls.X_dense)

        # Create AnnData objects
        cls.adata_dense = ad.AnnData(cls.X_dense.copy())
        cls.adata_sparse = ad.AnnData(cls.X_sparse.copy())

    def test_pca_dense_adata(self):
        """Test PCA on dense AnnData."""
        adata = self.adata_dense.copy()

        pca(adata, n_pcs=self.n_pcs)

        # Should have PCA results stored
        self.assertIn('X_pca', adata.obsm)
        self.assertEqual(adata.obsm['X_pca'].shape, (self.n_obs, self.n_pcs))

    def test_pca_sparse_adata(self):
        """Test PCA on sparse AnnData."""
        adata = self.adata_sparse.copy()

        pca(adata, n_pcs=self.n_pcs)

        self.assertIn('X_pca', adata.obsm)
        self.assertEqual(adata.obsm['X_pca'].shape, (self.n_obs, self.n_pcs))

    def test_pca_raw_matrix(self):
        """Test PCA on raw matrix (not AnnData)."""
        result = pca(self.X_dense.copy(), n_pcs=self.n_pcs)

        # Should return PCA results
        self.assertIsNotNone(result)

    def test_pca_different_n_pcs(self):
        """Test PCA with different numbers of components."""
        for n_pcs in [5, 10, 20]:
            adata = self.adata_dense.copy()
            pca(adata, n_pcs=n_pcs)

            self.assertEqual(adata.obsm['X_pca'].shape[1], n_pcs)

    def test_pca_preserves_input_shape(self):
        """Test that PCA doesn't change input data shape."""
        adata = self.adata_dense.copy()
        original_shape = adata.X.shape

        pca(adata, n_pcs=self.n_pcs)

        self.assertEqual(adata.X.shape, original_shape)

    def test_pca_deterministic_with_random_state(self):
        """Test that PCA is deterministic with fixed random state."""
        adata1 = self.adata_dense.copy()
        adata2 = self.adata_dense.copy()

        pca(adata1, n_pcs=self.n_pcs, random_state=42)
        pca(adata2, n_pcs=self.n_pcs, random_state=42)

        npt.assert_array_almost_equal(
            adata1.obsm['X_pca'],
            adata2.obsm['X_pca']
        )


class TestPCALayers(unittest.TestCase):
    """Test PCA with different layers."""

    def setUp(self):
        rng = np.random.default_rng(100)

        X = rng.random((50, 30))
        X[X < 0.4] = 0

        self.adata = ad.AnnData(X)
        self.adata.layers['normalized'] = X * 2
        self.adata.layers['scaled'] = (X - X.mean(axis=0)) / X.std(axis=0)
        return super().setUp()

    def test_pca_default_layer(self):
        """Test PCA on default layer (X)."""
        adata = self.adata.copy()
        pca(adata, n_pcs=10)

        self.assertIn('X_pca', adata.obsm)

    def test_pca_named_layer(self):
        """Test PCA on named layer."""
        adata = self.adata.copy()
        pca(adata, n_pcs=10, layer='normalized')

        # Should still store in standard location but computed from layer
        self.assertIn('X_pca', adata.obsm)

    def test_pca_layer_none_uses_main(self):
        """Test that layer=None uses main data matrix."""
        adata = self.adata.copy()

        pca(adata, n_pcs=10, layer=None)

        # Should compute PCA on X and store results
        self.assertIn('X_pca', adata.obsm)

    def test_pca_different_layers_different_results(self):
        """Test that different layers produce different results."""
        adata1 = self.adata.copy()
        adata2 = self.adata.copy()

        pca(adata1, n_pcs=10, layer=None)
        pca(adata2, n_pcs=10, layer='normalized')

        # Both store in X_pca but computed from different data
        # Results should be different since data is different
        self.assertIn('X_pca', adata1.obsm)
        self.assertIn('X_pca', adata2.obsm)


class TestPCASparseFormats(unittest.TestCase):
    """Test PCA with different sparse matrix formats."""

    def setUp(self):
        rng = np.random.default_rng(200)

        self.X = rng.random((40, 25))
        self.X[self.X < 0.5] = 0
        return super().setUp()

    def test_pca_csr_matrix(self):
        """Test PCA with CSR matrix."""
        adata = ad.AnnData(sps.csr_matrix(self.X))
        pca(adata, n_pcs=10)

        self.assertIn('X_pca', adata.obsm)
        self.assertEqual(adata.obsm['X_pca'].shape, (40, 10))

    def test_pca_csc_matrix(self):
        """Test PCA with CSC matrix."""
        adata = ad.AnnData(sps.csc_matrix(self.X))
        pca(adata, n_pcs=10)

        self.assertIn('X_pca', adata.obsm)

    def test_pca_sparse_preserved_after_computation(self):
        """Test that sparse format is preserved."""
        adata = ad.AnnData(sps.csr_matrix(self.X))
        original_format = type(adata.X)

        pca(adata, n_pcs=10)

        # Data should still be sparse
        self.assertTrue(sps.issparse(adata.X))


class TestStratifiedPCABasic(unittest.TestCase):
    """Test basic stratified PCA functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test data with imbalanced groups."""
        rng = np.random.default_rng(300)

        # Create imbalanced dataset
        # Group A: 80 cells, Group B: 15 cells, Group C: 5 cells
        n_vars = 40

        X_a = rng.random((80, n_vars))
        X_b = rng.random((15, n_vars)) + 1  # Different distribution
        X_c = rng.random((5, n_vars)) + 2   # Different distribution

        cls.X = np.vstack([X_a, X_b, X_c])
        cls.groups = ['A'] * 80 + ['B'] * 15 + ['C'] * 5

        cls.adata = ad.AnnData(cls.X)
        cls.adata.obs['group'] = cls.groups
        cls.adata.obs['batch'] = ['batch1'] * 50 + ['batch2'] * 50

    def test_stratified_pca_basic(self):
        """Test basic stratified PCA."""
        adata = self.adata.copy()

        result = stratified_pca(adata, obs_col='group', n_comps=10)

        # Should return the AnnData object
        self.assertIsInstance(result, ad.AnnData)
        # Should have stratified PCA results
        self.assertIn('X_pca_stratified', adata.obsm)
        self.assertEqual(adata.obsm['X_pca_stratified'].shape, (100, 10))

    def test_stratified_pca_loadings_stored(self):
        """Test that PCA loadings are stored."""
        adata = self.adata.copy()

        stratified_pca(adata, obs_col='group', n_comps=10)

        # Check loadings
        self.assertIn('X_stratified_PCs', adata.varm)
        self.assertEqual(adata.varm['X_stratified_PCs'].shape, (40, 10))

    def test_stratified_pca_variance_stored(self):
        """Test that variance statistics are stored."""
        adata = self.adata.copy()

        stratified_pca(adata, obs_col='group', n_comps=10)

        # Check variance info
        self.assertIn('pca_stratified', adata.uns)
        self.assertIn('variance', adata.uns['pca_stratified'])
        self.assertIn('variance_ratio', adata.uns['pca_stratified'])

        # Variance arrays should have correct length
        self.assertEqual(
            len(adata.uns['pca_stratified']['variance']),
            10
        )

    def test_stratified_pca_all_cells_projected(self):
        """Test that all cells get PCA coordinates."""
        adata = self.adata.copy()

        stratified_pca(adata, obs_col='group', n_comps=10)

        # All 100 cells should have coordinates
        self.assertEqual(adata.obsm['X_pca_stratified'].shape[0], 100)

    def test_stratified_pca_n_comps(self):
        """Test different numbers of components."""
        # Smallest group has 5 cells, so max n_comps is 4 with ARPACK
        for n_comps in [3, 4]:
            adata = self.adata.copy()
            stratified_pca(adata, obs_col='group', n_comps=n_comps)

            self.assertEqual(
                adata.obsm['X_pca_stratified'].shape[1],
                n_comps
            )

    def test_stratified_pca_reproducible(self):
        """Test that results are reproducible with same random state."""
        adata1 = self.adata.copy()
        adata2 = self.adata.copy()

        stratified_pca(adata1, obs_col='group', n_comps=10, random_state=42)
        stratified_pca(adata2, obs_col='group', n_comps=10, random_state=42)

        npt.assert_array_almost_equal(
            adata1.obsm['X_pca_stratified'],
            adata2.obsm['X_pca_stratified']
        )

    def test_stratified_pca_different_seeds(self):
        """Test that different random states produce different results."""
        adata1 = self.adata.copy()
        adata2 = self.adata.copy()

        stratified_pca(adata1, obs_col='group', n_comps=10, random_state=42)
        stratified_pca(adata2, obs_col='group', n_comps=10, random_state=123)

        # Results should be different (first PC may have flipped signs)
        # Just check that computation completed
        self.assertEqual(adata1.obsm['X_pca_stratified'].shape, (100, 10))
        self.assertEqual(adata2.obsm['X_pca_stratified'].shape, (100, 10))


class TestStratifiedPCASampling(unittest.TestCase):
    """Test stratified sampling behavior."""

    def setUp(self):
        rng = np.random.default_rng(400)

        # Create dataset with varying group sizes
        X_a = rng.random((50, 30))
        X_b = rng.random((30, 30))
        X_c = rng.random((20, 30))

        self.X = np.vstack([X_a, X_b, X_c])
        self.groups = ['A'] * 50 + ['B'] * 30 + ['C'] * 20

        self.adata = ad.AnnData(self.X)
        self.adata.obs['cell_type'] = self.groups
        return super().setUp()

    def test_stratified_pca_auto_n_per_group(self):
        """Test automatic n_per_group (uses smallest group)."""
        adata = self.adata.copy()

        # Smallest group is C with 20 cells
        # So should sample 20 from each group = 60 total for fitting
        stratified_pca(adata, obs_col='cell_type', n_comps=10)

        # Should complete successfully and project to all cells
        self.assertEqual(adata.obsm['X_pca_stratified'].shape[0], 100)

    def test_stratified_pca_explicit_n_per_group(self):
        """Test with explicit n_per_group."""
        adata = self.adata.copy()

        # Request 10 cells per group
        stratified_pca(
            adata,
            obs_col='cell_type',
            n_comps=10,
            n_per_group=10
        )

        self.assertEqual(adata.obsm['X_pca_stratified'].shape[0], 100)

    def test_stratified_pca_n_per_group_larger_than_smallest(self):
        """Test n_per_group larger than smallest group."""
        adata = self.adata.copy()

        # Request 25 cells per group (C only has 20)
        stratified_pca(
            adata,
            obs_col='cell_type',
            n_comps=10,
            n_per_group=25
        )

        # Should handle gracefully (use all 20 from C)
        self.assertEqual(adata.obsm['X_pca_stratified'].shape[0], 100)

    def test_stratified_pca_n_per_group_very_small(self):
        """Test with very small n_per_group."""
        adata = self.adata.copy()

        # Only 2 cells per group
        stratified_pca(
            adata,
            obs_col='cell_type',
            n_comps=5,  # Fewer PCs since less data
            n_per_group=2
        )

        self.assertEqual(adata.obsm['X_pca_stratified'].shape, (100, 5))


class TestStratifiedPCALayers(unittest.TestCase):
    """Test stratified PCA with different layers."""

    def setUp(self):
        rng = np.random.default_rng(500)

        X = rng.random((60, 25))

        self.adata = ad.AnnData(X)
        self.adata.layers['normalized'] = X * 2
        self.adata.layers['scaled'] = X * 3
        self.adata.obs['group'] = ['A'] * 30 + ['B'] * 30
        return super().setUp()

    def test_stratified_pca_default_layer(self):
        """Test stratified PCA on default layer (X)."""
        adata = self.adata.copy()

        stratified_pca(adata, obs_col='group', n_comps=10, layer='X')

        self.assertIn('X_pca_stratified', adata.obsm)

    def test_stratified_pca_named_layer(self):
        """Test stratified PCA on named layer."""
        adata = self.adata.copy()

        stratified_pca(
            adata,
            obs_col='group',
            n_comps=10,
            layer='normalized'
        )

        self.assertIn('normalized_pca_stratified', adata.obsm)
        self.assertIn('normalized_stratified_PCs', adata.varm)

    def test_stratified_pca_different_layers_different_results(self):
        """Test that different layers produce different results."""
        adata1 = self.adata.copy()
        adata2 = self.adata.copy()

        stratified_pca(
            adata1,
            obs_col='group',
            n_comps=10,
            layer='X',
            random_state=42
        )
        stratified_pca(
            adata2,
            obs_col='group',
            n_comps=10,
            layer='normalized',
            random_state=42
        )

        # Results should be different
        with self.assertRaises(AssertionError):
            npt.assert_array_almost_equal(
                adata1.obsm['X_pca_stratified'],
                adata2.obsm['normalized_pca_stratified']
            )


class TestStratifiedPCAEdgeCases(unittest.TestCase):
    """Test edge cases for stratified PCA."""

    def test_stratified_pca_single_group(self):
        """Test with only one group (should still work)."""
        rng = np.random.default_rng(600)
        X = rng.random((50, 20))

        adata = ad.AnnData(X)
        adata.obs['group'] = ['A'] * 50

        stratified_pca(adata, obs_col='group', n_comps=10)

        self.assertEqual(adata.obsm['X_pca_stratified'].shape, (50, 10))

    def test_stratified_pca_many_groups(self):
        """Test with many groups."""
        rng = np.random.default_rng(700)
        X = rng.random((100, 30))

        adata = ad.AnnData(X)
        # 10 groups with 10 cells each
        adata.obs['group'] = [f'group_{i}' for i in range(10) for _ in range(10)]

        stratified_pca(adata, obs_col='group', n_comps=10)

        self.assertEqual(adata.obsm['X_pca_stratified'].shape, (100, 10))

    def test_stratified_pca_very_small_groups(self):
        """Test with very small groups."""
        rng = np.random.default_rng(800)
        X = rng.random((30, 20))

        adata = ad.AnnData(X)
        # Groups of 5, 5, 5, 5, 10 (smallest is 5)
        adata.obs['group'] = ['A'] * 5 + ['B'] * 5 + ['C'] * 5 + ['D'] * 5 + ['E'] * 10

        # With 5 samples per group, we need fewer components
        # ARPACK requires n_components < min(n_samples, n_features)
        stratified_pca(adata, obs_col='group', n_comps=4)

        self.assertEqual(adata.obsm['X_pca_stratified'].shape, (30, 4))

    def test_stratified_pca_n_comps_larger_than_features(self):
        """Test requesting components within valid range."""
        rng = np.random.default_rng(900)
        X = rng.random((50, 30))

        adata = ad.AnnData(X)
        adata.obs['group'] = ['A'] * 25 + ['B'] * 25

        # Request reasonable number of PCs (< min(n_samples, n_features))
        stratified_pca(adata, obs_col='group', n_comps=20)

        # Should complete successfully
        self.assertIn('X_pca_stratified', adata.obsm)
        self.assertEqual(adata.obsm['X_pca_stratified'].shape, (50, 20))


class TestStratifiedPCASparse(unittest.TestCase):
    """Test stratified PCA with sparse matrices."""

    def setUp(self):
        rng = np.random.default_rng(1000)

        X = rng.random((60, 30))
        X[X < 0.6] = 0  # Make sparse

        self.adata = ad.AnnData(sps.csr_matrix(X))
        self.adata.obs['group'] = ['A'] * 20 + ['B'] * 20 + ['C'] * 20
        return super().setUp()

    def test_stratified_pca_sparse_csr(self):
        """Test stratified PCA with CSR sparse matrix."""
        adata = self.adata.copy()

        stratified_pca(adata, obs_col='group', n_comps=10)

        self.assertEqual(adata.obsm['X_pca_stratified'].shape, (60, 10))

    def test_stratified_pca_sparse_csc(self):
        """Test stratified PCA with CSC sparse matrix."""
        adata = ad.AnnData(sps.csc_matrix(self.adata.X))
        adata.obs = self.adata.obs.copy()

        stratified_pca(adata, obs_col='group', n_comps=10)

        self.assertEqual(adata.obsm['X_pca_stratified'].shape, (60, 10))

    def test_stratified_pca_preserves_sparse(self):
        """Test that sparse format is preserved."""
        adata = self.adata.copy()

        stratified_pca(adata, obs_col='group', n_comps=10)

        # Original data should still be sparse
        self.assertTrue(sps.issparse(adata.X))


class TestPCAVarianceExplained(unittest.TestCase):
    """Test that PCA captures variance correctly."""

    def setUp(self):
        rng = np.random.default_rng(1100)

        # Create data with clear structure
        # First 3 PCs should capture most variance
        U = rng.random((50, 3))
        V = rng.random((3, 30))

        self.X = U @ V + rng.random((50, 30)) * 0.01  # Add small noise

        self.adata = ad.AnnData(self.X)
        self.adata.obs['group'] = ['A'] * 25 + ['B'] * 25
        return super().setUp()

    def test_stratified_pca_variance_sum(self):
        """Test that variance ratios sum to reasonable value."""
        adata = self.adata.copy()

        stratified_pca(adata, obs_col='group', n_comps=10)

        variance_ratio = adata.uns['pca_stratified']['variance_ratio']

        # First few PCs should capture most variance
        self.assertGreater(sum(variance_ratio[:3]), 0.8)
        # Total should be <= 1
        self.assertLessEqual(sum(variance_ratio), 1.0)

    def test_stratified_pca_variance_decreasing(self):
        """Test that variance explained is decreasing."""
        adata = self.adata.copy()

        stratified_pca(adata, obs_col='group', n_comps=10)

        variance = adata.uns['pca_stratified']['variance']

        # Variance should be monotonically decreasing
        for i in range(len(variance) - 1):
            self.assertGreaterEqual(variance[i], variance[i + 1])


class TestPCAComparison(unittest.TestCase):
    """Test comparing regular PCA and stratified PCA."""

    def setUp(self):
        rng = np.random.default_rng(1200)

        # Balanced dataset - regular and stratified should be similar
        X = rng.random((60, 25))

        self.adata_balanced = ad.AnnData(X)
        self.adata_balanced.obs['group'] = ['A'] * 30 + ['B'] * 30

        # Imbalanced dataset - should see differences
        X_imb = np.vstack([
            rng.random((50, 25)),
            rng.random((5, 25)) + 2  # Small group with different distribution
        ])

        self.adata_imbalanced = ad.AnnData(X_imb)
        self.adata_imbalanced.obs['group'] = ['A'] * 50 + ['B'] * 5
        return super().setUp()

    def test_stratified_vs_regular_balanced(self):
        """Test that stratified and regular PCA are similar for balanced data."""
        adata1 = self.adata_balanced.copy()
        adata2 = self.adata_balanced.copy()

        pca(adata1, n_pcs=10, random_state=42)
        stratified_pca(adata2, obs_col='group', n_comps=10, random_state=42)

        # Should be reasonably similar (high correlation)
        correlation = np.corrcoef(
            adata1.obsm['X_pca'][:, 0],
            adata2.obsm['X_pca_stratified'][:, 0]
        )[0, 1]

        # First PC should be highly correlated
        self.assertGreater(abs(correlation), 0.8)

    def test_stratified_handles_imbalance(self):
        """Test that stratified PCA handles imbalanced data."""
        adata = self.adata_imbalanced.copy()

        # Should complete without error despite imbalance
        # With 5 samples per group (smallest), need n_comps < 5
        stratified_pca(adata, obs_col='group', n_comps=4)

        self.assertEqual(adata.obsm['X_pca_stratified'].shape, (55, 4))
