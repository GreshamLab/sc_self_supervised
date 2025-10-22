import unittest
import numpy as np
import numpy.testing as npt
import anndata as ad
import scipy.sparse as sps
from sklearn.preprocessing import MinMaxScaler

from scself.utils import standardize_data, array_sum
from scself import TruncRobustScaler, TruncMinMaxScaler

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
        x = x.toarray()
    if sps.issparse(y):
        y = y.toarray()

    npt.assert_array_almost_equal(
        x,
        y
    )


class TestScalingDense(unittest.TestCase):

    def setUp(self) -> None:
        self.data = ad.AnnData(X.copy())
        self.data.layers['a'] = X.copy()
        self.data.obs['strat'] = ['A', 'B'] * 50

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

    def test_depth_cap(self):

        standardize_data(self.data, target_sum=50, method='depth', size_factor_cap=1)

        _sf = np.clip(SF, 1, None)

        _equal(
            np.divide(X, _sf[:, None]),
            self.data.X
        )
        _equal(
            _sf,
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

    def test_depth_stratified_equal(self):

        standardize_data(
            self.data,
            target_sum={'A': 50, 'B': 50},
            stratification_column='strat',
            method='depth'
        )
        _equal(
            np.divide(X, SF[:, None]),
            self.data.X
        )
        _equal(
            SF,
            self.data.obs['X_size_factor'].values
        )

    def test_depth_stratified_equal_sampling(self):

        standardize_data(
            self.data,
            target_sum={'A': 50, 'B': 50},
            stratification_column='strat',
            method='depth',
            depth_by_sampling=True
        )
        _equal(
            np.full(self.data.shape[0], 50),
            array_sum(self.data.X, 1)
        )
        _equal(
            SF,
            self.data.obs['X_size_factor'].values
        )

    def test_depth_stratified_unequal(self):

        standardize_data(
            self.data,
            target_sum={'A': 50, 'B': 25},
            stratification_column='strat',
            method='depth'
        )

        _sf = COUNT / np.tile([50, 25], 50) 

        _equal(
            np.divide(X, _sf[:, None]),
            self.data.X
        )
        _equal(
            _sf,
            self.data.obs['X_size_factor'].values
        )

    def test_depth_stratified_unequal_sampling(self):

        standardize_data(
            self.data,
            target_sum={'A': 50, 'B': 25},
            stratification_column='strat',
            method='depth',
            depth_by_sampling=True
        )
        _equal(
            np.tile([50, 25], 50),
            array_sum(self.data.X, 1)
        )
        _equal(
            COUNT / np.tile([50, 25], 50) ,
            self.data.obs['X_size_factor'].values
        )

    def test_depth_stratified(self):

        standardize_data(
            self.data,
            stratification_column='strat',
            method='depth'
        )

        _sf = COUNT / np.tile(
            [np.median(COUNT[::2]), np.median(COUNT[1::2])],
            50
        ) 

        _equal(
            np.divide(X, _sf[:, None]),
            self.data.X
        )
        _equal(
            _sf,
            self.data.obs['X_size_factor'].values
        )

    def test_depth_stratified_sampling(self):

        standardize_data(
            self.data,
            stratification_column='strat',
            method='depth',
            depth_by_sampling=True
        )

        _targets = [np.median(COUNT[::2]), np.median(COUNT[1::2])]

        _equal(
            np.tile(_targets, 50),
            array_sum(self.data.X, 1)
        )
        _equal(
            COUNT / np.tile(_targets, 50),
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


class TestMinMaxScaling(unittest.TestCase):

    def setUp(self) -> None:
        self.data = ad.AnnData(X.copy())
        return super().setUp()
    
    def test_scale_no_trunc(self):

        scaler = TruncMinMaxScaler(quantile_range=(0.0, 1.0)).fit(X)
        other_scaler = MinMaxScaler().fit(X)

        npt.assert_almost_equal(scaler.scale_, other_scaler.scale_)
        npt.assert_almost_equal(scaler.min_, other_scaler.min_)

        npt.assert_almost_equal(
            scaler.transform(X),
            other_scaler.transform(X)
        )


    def test_scale_trunc(self):

        scaler = TruncMinMaxScaler(quantile_range=(None, 0.8)).fit(X)

        for i in range(X.shape[1]):
            self.data.X[:, i] = np.clip(
                self.data.X[:, i],
                0,
                np.nanquantile(self.data.X[:, i], 0.8, method='higher')
            )

        npt.assert_equal(
            np.max(self.data.X, axis=0),
            scaler.data_range_
        )

        other_scaler = MinMaxScaler().fit(self.data.X)

        npt.assert_almost_equal(scaler.min_, other_scaler.min_)
        npt.assert_almost_equal(scaler.scale_, other_scaler.scale_)

        npt.assert_almost_equal(
            scaler.transform(X),
            other_scaler.transform(self.data.X)
        )


    def test_scale_trunc_twoside(self):

        scaler = TruncMinMaxScaler(quantile_range=(0.2, 0.8)).fit(X)

        self.assertEqual(X.shape[1], scaler.scale_.shape[0])
        self.assertEqual(X.shape[1], scaler.min_.shape[0])

        for i in range(X.shape[1]):
            self.data.X[:, i] = np.clip(
                self.data.X[:, i],
                np.nanquantile(self.data.X[:, i], 0.2, method='lower'),
                np.nanquantile(self.data.X[:, i], 0.8, method='higher')
            )

        npt.assert_equal(
            np.max(self.data.X, axis=0) - np.min(self.data.X, axis=0),
            scaler.data_range_
        )

        other_scaler = MinMaxScaler().fit(self.data.X)

        npt.assert_almost_equal(scaler.min_, other_scaler.min_)
        npt.assert_almost_equal(scaler.scale_, other_scaler.scale_)

        npt.assert_almost_equal(
            scaler.transform(X),
            other_scaler.transform(self.data.X)
        )

    def test_scale_trunc_explicit_clip(self):

        scaler = TruncMinMaxScaler(quantile_range=None, clipping_range=(1, 3)).fit(X)

        self.assertEqual(X.shape[1], scaler.scale_.shape[0])
        self.assertEqual(X.shape[1], scaler.min_.shape[0])

        for i in range(X.shape[1]):
            self.data.X[:, i] = np.clip(
                self.data.X[:, i],
                1,
                3
            )

        npt.assert_equal(
            np.max(self.data.X, axis=0) - np.min(self.data.X, axis=0),
            scaler.data_range_
        )

        other_scaler = MinMaxScaler().fit(self.data.X)

        npt.assert_almost_equal(scaler.min_, other_scaler.min_)
        npt.assert_almost_equal(scaler.scale_, other_scaler.scale_)

        npt.assert_almost_equal(
            scaler.transform(X),
            other_scaler.transform(self.data.X)
        )



class TestScalingCSR(TestScalingDense):

    def setUp(self) -> None:
        self.data = ad.AnnData(sps.csr_matrix(X))
        self.data.layers['a'] = sps.csr_matrix(X)
        self.data.obs['strat'] = ['A', 'B'] * 50


class TestScalingCSC(TestScalingDense):

    def setUp(self) -> None:
        self.data = ad.AnnData(sps.csc_matrix(X))
        self.data.layers['a'] = sps.csc_matrix(X)
        self.data.obs['strat'] = ['A', 'B'] * 50

    @unittest.skip
    def test_depth_stratified_equal_sampling(self):
        pass

    @unittest.skip
    def test_depth_stratified_unequal_sampling(self):
        pass

    @unittest.skip
    def test_depth_stratified_sampling(self):
        pass


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self) -> None:
        self.data = ad.AnnData(X.copy())
        self.data.obs['strat'] = ['A', 'B'] * 50
        return super().setUp()

    def test_empty_cells(self):
        """Test normalization with cells that have zero counts."""
        data_with_zeros = self.data.copy()
        data_with_zeros.X[0, :] = 0
        data_with_zeros.X[5, :] = 0

        standardize_data(data_with_zeros, target_sum=50, method='log')

        # Cells with zero counts should have size factor of 1
        self.assertEqual(data_with_zeros.obs['X_size_factor'].values[0], 1.0)
        self.assertEqual(data_with_zeros.obs['X_size_factor'].values[5], 1.0)
        # Empty cells should remain zero after normalization
        _equal(data_with_zeros.X[0, :], np.zeros(X.shape[1]))
        _equal(data_with_zeros.X[5, :], np.zeros(X.shape[1]))

    def test_median_target_sum(self):
        """Test that target_sum=None uses median of counts."""
        data = self.data.copy()
        standardize_data(data, target_sum=None, method='depth')

        expected_target = np.median(COUNT)
        self.assertAlmostEqual(
            data.obs['X_target_sum'].values[0],
            expected_target
        )

    def test_standardization_metadata(self):
        """Test that standardization parameters are stored in uns."""
        data = self.data.copy()
        standardize_data(
            data,
            target_sum={'A': 100, 'B': 100},
            method='log_scale',
            stratification_column='strat',
            size_factor_cap=2.0,
            depth_by_sampling=False,
            random_state=42
        )

        self.assertIn('standardization', data.uns)
        std_params = data.uns['standardization']
        self.assertTrue(std_params['log'])
        self.assertTrue(std_params['scale'])
        # target_sum will be an array when stratified
        self.assertEqual(std_params['stratification_column'], 'strat')
        self.assertEqual(std_params['size_factor_cap'], 2.0)
        self.assertFalse(std_params['depth_by_sampling'])
        self.assertEqual(std_params['random_state'], 42)

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        with self.assertRaises(ValueError) as context:
            standardize_data(self.data, method='invalid_method')

        self.assertIn('invalid_method', str(context.exception))

    def test_stratification_without_adata_raises_error(self):
        """Test that stratification requires proper parameters."""
        from scself.utils.standardization import size_factors

        with self.assertRaises(ValueError) as context:
            size_factors(
                X,
                target_sum=50,
                adata=None,
                stratification_col='strat'
            )

        self.assertIn('adata', str(context.exception))

    def test_stratification_target_sum_type_error(self):
        """Test that stratified normalization requires dict/Series target_sum."""
        from scself.utils.standardization import size_factors

        with self.assertRaises(ValueError) as context:
            size_factors(
                X,
                target_sum=50,  # Should be dict or Series
                adata=self.data,
                stratification_col='strat'
            )

        self.assertIn('dict or pd.Series', str(context.exception))

    def test_sampling_with_csc_raises_error(self):
        """Test that sampling with CSC format raises error."""
        from scself.utils.standardization import _normalize_by_sampling

        csc_data = sps.csc_matrix(X)

        with self.assertRaises(RuntimeError) as context:
            _normalize_by_sampling(csc_data, target_sum=50)

        self.assertIn('CSR or dense', str(context.exception))

    def test_size_factor_cap_applied(self):
        """Test that size_factor_cap properly limits size factors."""
        data = self.data.copy()
        standardize_data(
            data,
            target_sum=10,  # Low target to create large size factors
            method='depth',
            size_factor_cap=1.5
        )

        # All size factors should be >= 1.5
        self.assertTrue(np.all(data.obs['X_size_factor'].values >= 1.5))

    def test_scale_factor_return_value(self):
        """Test that scale methods return scale factors."""
        data = self.data.copy()
        _, scale_factor = standardize_data(data, target_sum=50, method='scale')

        self.assertIsNotNone(scale_factor)
        self.assertEqual(scale_factor.shape[0], X.shape[1])

    def test_no_scale_returns_none(self):
        """Test that non-scaling methods return None for scale_factor."""
        data = self.data.copy()
        _, scale_factor = standardize_data(data, target_sum=50, method='log')

        self.assertIsNone(scale_factor)

    def test_warnings_for_conflicting_parameters(self):
        """Test that warnings are issued for conflicting parameters."""
        data = self.data.copy()

        # Test warning when size_factor conflicts with target_sum
        with self.assertWarns(UserWarning):
            standardize_data(
                data,
                target_sum=50,
                size_factor=np.ones(100),
                method='depth'
            )

    def test_warnings_for_sampling_conflicts(self):
        """Test warnings when sampling conflicts with size_factor parameters."""
        data = self.data.copy()

        with self.assertWarns(UserWarning):
            standardize_data(
                data,
                target_sum=50,
                size_factor=np.ones(100),
                depth_by_sampling=True,
                method='depth'
            )


class TestSamplingNormalization(unittest.TestCase):
    """Test sampling-based normalization in detail."""

    def setUp(self) -> None:
        self.data = ad.AnnData(X.copy())
        return super().setUp()

    def test_sampling_produces_exact_counts(self):
        """Test that sampling produces exactly target_sum counts."""
        data = self.data.copy()
        target = 100
        standardize_data(
            data,
            target_sum=target,
            method='depth',
            depth_by_sampling=True
        )

        counts = array_sum(data.X, 1)
        _equal(counts, np.full(data.shape[0], target))

    def test_sampling_with_array_target(self):
        """Test sampling with different target for each cell."""
        from scself.utils.standardization import _normalize_by_sampling

        data = X.copy().astype(float)
        targets = np.arange(50, 150)  # Different target for each cell

        _normalize_by_sampling(data, target_sum=targets, random_state=42)

        counts = array_sum(data, 1)
        _equal(counts, targets)

    def test_sampling_reproducibility(self):
        """Test that sampling is reproducible with same random_state."""
        data1 = self.data.copy()
        data2 = self.data.copy()

        standardize_data(
            data1,
            target_sum=50,
            method='depth',
            depth_by_sampling=True,
            random_state=123
        )
        standardize_data(
            data2,
            target_sum=50,
            method='depth',
            depth_by_sampling=True,
            random_state=123
        )

        _equal(data1.X, data2.X)

    def test_sampling_different_seeds(self):
        """Test that different random states produce different results."""
        data1 = self.data.copy()
        data2 = self.data.copy()

        standardize_data(
            data1,
            target_sum=50,
            method='depth',
            depth_by_sampling=True,
            random_state=123
        )
        standardize_data(
            data2,
            target_sum=50,
            method='depth',
            depth_by_sampling=True,
            random_state=456
        )

        # Results should be different (with high probability)
        self.assertFalse(np.allclose(data1.X, data2.X))


class TestHelperFunctions(unittest.TestCase):
    """Test internal helper functions."""

    def test_get_layer_main(self):
        """Test getting main X layer."""
        from scself.utils.standardization import _get_layer

        data = ad.AnnData(X.copy())
        layer_data = _get_layer(data, 'X')

        _equal(layer_data, X)

    def test_get_layer_named(self):
        """Test getting named layer."""
        from scself.utils.standardization import _get_layer

        data = ad.AnnData(X.copy())
        data.layers['test'] = X * 2
        layer_data = _get_layer(data, 'test')

        _equal(layer_data, X * 2)

    def test_log1p_dense(self):
        """Test log1p transformation on dense array."""
        from scself.utils.standardization import log1p

        data = X.copy().astype(float)
        expected = np.log1p(X)

        log1p(data)
        _equal(data, expected)

    def test_log1p_sparse(self):
        """Test log1p transformation on sparse matrix."""
        from scself.utils.standardization import log1p

        data = sps.csr_matrix(X.astype(float))
        expected = np.log1p(X)

        log1p(data)
        _equal(data, expected)

    def test_normalize_total_without_size_factor(self):
        """Test _normalize_total computes size factors when not provided."""
        from scself.utils.standardization import _normalize_total, _size_factors_all

        data = X.copy()
        # _size_factors_all returns (counts, size_factor, target_sum)
        _, expected_sf, _ = _size_factors_all(data, target_sum=50)

        result = _normalize_total(data, target_sum=50)

        # Should normalize using computed size factors
        self.assertIsNotNone(result)
        # Verify the result is normalized correctly
        _equal(result, X / expected_sf[:, None])

    def test_normalize_total_with_size_factor(self):
        """Test _normalize_total uses provided size factors."""
        from scself.utils.standardization import _normalize_total

        data = X.copy()
        size_factors = np.ones(X.shape[0]) * 2
        result = _normalize_total(data, size_factor=size_factors)

        # Should divide by 2
        _equal(result, X / 2)

    def test_size_factors_all_with_median(self):
        """Test _size_factors_all computes correct median target."""
        from scself.utils.standardization import _size_factors_all

        counts, sf, target = _size_factors_all(X, target_sum=None)

        _equal(counts, COUNT)
        self.assertAlmostEqual(target, np.median(COUNT))
        _equal(sf, counts / target)

    def test_size_factors_all_with_target(self):
        """Test _size_factors_all with explicit target."""
        from scself.utils.standardization import _size_factors_all

        counts, sf, target = _size_factors_all(X, target_sum=100)

        _equal(counts, COUNT)
        self.assertEqual(target, 100)
        _equal(sf, counts / 100)

    def test_size_factors_stratified_with_median(self):
        """Test stratified size factors compute group medians."""
        from scself.utils.standardization import _size_factors_stratified

        data = ad.AnnData(X)
        data.obs['group'] = ['A', 'B'] * 50

        counts, sf, targets = _size_factors_stratified(
            X,
            data,
            'group',
            target_sum=None
        )

        _equal(counts, COUNT)
        # Check that targets alternate between group medians
        self.assertEqual(targets[0], targets[2])
        self.assertEqual(targets[1], targets[3])
        self.assertNotEqual(targets[0], targets[1])


class TestSubsetGenesForDepth(unittest.TestCase):
    """Test subset genes for depth calculation."""

    def setUp(self) -> None:
        self.data = ad.AnnData(X.copy())
        return super().setUp()

    def test_subset_genes_boolean_mask(self):
        """Test subset_genes_for_depth with boolean mask."""
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[0:5] = True

        standardize_data(
            self.data,
            target_sum=20,
            method='depth',
            subset_genes_for_depth=mask
        )

        # Size factors should be based on first 5 genes
        subset_counts = X[:, 0:5].sum(1)
        sf = subset_counts / 20
        sf[sf == 0] = 1.

        _equal(self.data.obs['X_size_factor'].values, sf)
        _equal(self.data.obs['X_subset_counts'].values, subset_counts)
        _equal(self.data.obs['X_counts'].values, COUNT)

    def test_subset_genes_with_layer(self):
        """Test subset genes with named layer."""
        self.data.layers['test'] = X.copy() * 2

        standardize_data(
            self.data,
            target_sum=40,
            method='depth',
            subset_genes_for_depth=['0', '1', '2', '3', '4'],
            layer='test'
        )

        # Check that X is unchanged
        _equal(self.data.X, X)
        # Check that test layer was normalized
        self.assertFalse(np.allclose(self.data.layers['test'], X * 2))


class TestMultipleLayerNormalization(unittest.TestCase):
    """Test normalization of multiple layers."""

    def setUp(self) -> None:
        self.data = ad.AnnData(X.copy())
        self.data.layers['raw'] = X.copy()
        self.data.layers['processed'] = X.copy()
        return super().setUp()

    def test_independent_layer_normalization(self):
        """Test that different layers can be normalized independently."""
        # Normalize main layer
        standardize_data(self.data, target_sum=50, method='log', layer='X')

        # Normalize another layer with different parameters
        standardize_data(self.data, target_sum=100, method='scale', layer='raw')

        # Check that layers are different
        self.assertFalse(np.allclose(self.data.X, self.data.layers['raw']))
        # Check that processed layer is still unchanged
        _equal(self.data.layers['processed'], X)

    def test_layer_metadata_separation(self):
        """Test that each layer gets separate metadata."""
        standardize_data(self.data, target_sum=50, method='log', layer='X')
        standardize_data(self.data, target_sum=100, method='log', layer='raw')

        self.assertIn('X_size_factor', self.data.obs)
        self.assertIn('raw_size_factor', self.data.obs)
        self.assertIn('X_counts', self.data.obs)
        self.assertIn('raw_counts', self.data.obs)

        # Size factors should be different
        self.assertFalse(
            np.allclose(
                self.data.obs['X_size_factor'].values,
                self.data.obs['raw_size_factor'].values
            )
        )


class TestSparseMatrixFormats(unittest.TestCase):
    """Test different sparse matrix formats."""

    def test_csr_maintains_format(self):
        """Test that CSR format is maintained after normalization."""
        data = ad.AnnData(sps.csr_matrix(X))
        standardize_data(data, target_sum=50, method='depth')

        self.assertTrue(sps.issparse(data.X))
        self.assertTrue(sps.isspmatrix_csr(data.X))

    def test_csc_converted_for_operations(self):
        """Test that CSC format is handled correctly."""
        data = ad.AnnData(sps.csc_matrix(X))
        standardize_data(data, target_sum=50, method='log')

        # Should still produce correct result
        self.assertTrue(sps.issparse(data.X))

    def test_dense_remains_dense(self):
        """Test that dense arrays remain dense."""
        data = ad.AnnData(X.copy())
        standardize_data(data, target_sum=50, method='log')

        self.assertFalse(sps.issparse(data.X))
        self.assertIsInstance(data.X, np.ndarray)


class TestDataTypeHandling(unittest.TestCase):
    """Test handling of different data types."""

    def test_integer_input_converted_to_float(self):
        """Test that integer input is converted to float during normalization."""
        data = ad.AnnData(X.astype(int))
        self.assertEqual(data.X.dtype, int)

        standardize_data(data, target_sum=50, method='depth')

        # After normalization, should be float
        self.assertTrue(np.issubdtype(data.X.dtype, np.floating))

    def test_float_input_maintained(self):
        """Test that float input remains float."""
        data = ad.AnnData(X.astype(float))
        standardize_data(data, target_sum=50, method='depth')

        self.assertTrue(np.issubdtype(data.X.dtype, np.floating))


class TestScaleFactorPersistence(unittest.TestCase):
    """Test that scale factors are correctly stored and can be reused."""

    def setUp(self) -> None:
        self.data1 = ad.AnnData(X.copy())
        self.data2 = ad.AnnData(X.copy())
        return super().setUp()

    def test_reuse_scale_factor(self):
        """Test that scale factors can be extracted and reused."""
        # First normalization - compute scale factors
        _, scale_factors = standardize_data(
            self.data1,
            target_sum=50,
            method='scale'
        )

        # Second normalization - reuse scale factors
        standardize_data(
            self.data2,
            target_sum=50,
            method='scale',
            scale_factor=scale_factors
        )

        # Results should be identical
        _equal(self.data1.X, self.data2.X)
        _equal(
            self.data1.var['X_scale_factor'].values,
            self.data2.var['X_scale_factor'].values
        )

    def test_reuse_size_factor(self):
        """Test that size factors can be extracted and reused."""
        # First normalization
        standardize_data(self.data1, target_sum=50, method='depth')
        size_factors = self.data1.obs['X_size_factor'].values.copy()

        # Second normalization - reuse size factors
        standardize_data(
            self.data2,
            size_factor=size_factors,
            method='depth'
        )

        # Results should be identical
        _equal(self.data1.X, self.data2.X)
        _equal(
            self.data1.obs['X_size_factor'].values,
            self.data2.obs['X_size_factor'].values
        )
