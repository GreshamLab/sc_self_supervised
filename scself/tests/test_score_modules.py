"""
Generated with cursor/claude-3.5-sonnet and then fixed to actually work
"""


import unittest
import numpy as np
import scipy.sparse as sps
import anndata as ad
import numpy.testing as npt
from scself.scaling import TruncMinMaxScaler
from scself._modules.score_modules import (
    score_all_modules,
    module_score,
    regress_out_to_residuals
)

class TestModuleScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test data
        np.random.seed(42)
        n_obs = 100
        n_vars = 50
        
        # Create expression matrix with some clear patterns
        cls.X = np.random.negative_binomial(20, 0.3, (n_obs, n_vars))
        cls.n_counts = cls.X.sum(axis=1)
        
        # Create module assignments
        cls.modules = np.repeat(['A', 'B', 'C'], n_vars // 3 + 2)[:n_vars]
        
        # Create AnnData object
        cls.adata = ad.AnnData(
            X=cls.X,
            var={"gene_module": cls.modules}
        )
        
        # Create sparse version
        cls.adata_sparse = ad.AnnData(
            X=sps.csr_matrix(cls.X),
            var={"gene_module": cls.modules}
        )

    def test_regress_out_to_residuals(self):

        scores = module_score(
            self.adata,
            self.adata.var_names
        )

        resids = regress_out_to_residuals(
            self.n_counts,
            scores
        )

        from scipy.stats import linregress

        _regress = linregress(
            self.n_counts,
            scores
        )
        _yhat = self.n_counts * _regress.slope + _regress.intercept
        _test_resids = scores - _yhat

        npt.assert_allclose(
            _test_resids.reshape(-1, 1),
            resids
        )

    def test_basic_scoring(self):
        """Test basic module scoring functionality"""
        result = score_all_modules(self.adata.copy())
        
        self.assertIn('gene_module_score', result.obsm)
        self.assertIn('gene_module_score_corrcoef', result.uns)
        self.assertEqual(result.obsm['gene_module_score'].shape[1], 3)  # A, B, C modules
        
    def test_sparse_scoring(self):
        """Test module scoring with sparse input"""
        result = score_all_modules(self.adata_sparse.copy())
        
        dense_result = score_all_modules(self.adata.copy())
        npt.assert_allclose(
            result.obsm['gene_module_score'],
            dense_result.obsm['gene_module_score'],
            rtol=1e-5
        )

    def test_sparse_scoring_regress_counts(self):
        """Test module scoring with sparse input"""

        result = score_all_modules(self.adata_sparse.copy(), regress_out_variable=self.n_counts)
        dense_result = score_all_modules(self.adata.copy(), regress_out_variable=self.n_counts)

        npt.assert_allclose(
            result.obsm['gene_module_score'],
            dense_result.obsm['gene_module_score'],
            rtol=1e-5
        )

    def test_custom_modules(self):
        """Test scoring with explicitly provided modules"""
        custom_modules = ['A', 'C']  # Skip module B
        result = score_all_modules(self.adata.copy(), modules=custom_modules)
        
        self.assertEqual(result.obsm['gene_module_score'].shape[1], 2)
        self.assertEqual(len(result.uns['gene_module_score']), 2)

    def test_obs_mask(self):
        """Test scoring with observation mask"""
        mask = np.zeros(self.adata.n_obs, dtype=bool)
        mask[::2] = True  # Score every other cell
        
        result = score_all_modules(self.adata.copy(), obs_mask=mask)
        
        # Check that masked values are NaN
        self.assertTrue(np.all(np.isnan(result.obsm['gene_module_score'][~mask])))
        self.assertTrue(np.all(~np.isnan(result.obsm['gene_module_score'][mask])))

    def test_custom_layer(self):
        """Test scoring with custom layer"""
        adata = self.adata.copy()
        adata.layers['custom'] = np.random.negative_binomial(20, 0.3, adata.shape)
        
        result = score_all_modules(adata, layer='custom')
        base_result = score_all_modules(adata.copy())
        
        # Scores should be different due to different input values
        self.assertTrue(np.any(np.not_equal(
            result.obsm['gene_module_score'],
            base_result.obsm['gene_module_score']
        )))

    def test_empty_module(self):
        """Test handling of empty modules"""
        adata = self.adata.copy()
        adata.var['gene_module'] = 'A'  # Make all genes module A
        
        result = score_all_modules(adata, modules=['A', 'B'])
        
        # Check B module scores are all NaN
        self.assertTrue(np.all(np.isnan(result.obsm['gene_module_score'][:, 1])))
        
    def test_single_gene_module(self):
        """Test scoring module with single gene"""
        adata = self.adata.copy()
        adata.var['gene_module'] = ['A'] + ['B'] * (adata.shape[1] - 1)
        
        result = score_all_modules(adata)
        
        # Check scores exist for both modules
        self.assertEqual(result.obsm['gene_module_score'].shape[1], 2)

    def test_custom_scaler(self):
        """Test with custom scaler settings"""
        result = score_all_modules(
            self.adata.copy(),
            scaler=TruncMinMaxScaler(quantile_range=(0.1, 0.9))
        )
        
        # Scores should be between 0 and 1
        self.assertTrue(np.all(result.obsm['gene_module_score'] >= 0))
        self.assertTrue(np.all(result.obsm['gene_module_score'] <= 1))

    def test_no_scaling(self):
        """Test without any scaling"""
        result = score_all_modules(self.adata.copy(), scaler=None)
        
        # Raw scores should be different from scaled scores
        base_result = score_all_modules(self.adata.copy())
        self.assertTrue(np.any(np.not_equal(
            result.obsm['gene_module_score'],
            base_result.obsm['gene_module_score']
        )))

    def test_custom_clipping(self):
        """Test with custom clipping values"""
        clip_range = (-5, 5)
        result = score_all_modules(self.adata.copy(), clipping=clip_range)
        
        # Check scores are within clipping range
        self.assertTrue(np.all(result.obsm['gene_module_score'] >= clip_range[0]))
        self.assertTrue(np.all(result.obsm['gene_module_score'] <= clip_range[1]))

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        # Test invalid layer
        with self.assertRaises(KeyError):
            score_all_modules(self.adata.copy(), layer='nonexistent')
            
        # Test invalid module column
        with self.assertRaises(KeyError):
            score_all_modules(self.adata.copy(), module_column='nonexistent')
            
        # Test invalid module list
        result = score_all_modules(self.adata.copy(), modules=['nonexistent']) 

        self.assertTrue(np.all(np.isnan(result.obsm['gene_module_score'])))
