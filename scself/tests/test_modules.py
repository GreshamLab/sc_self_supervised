"""
Generated with cursor/claude-3.5-sonnet and then fixed to actually work
"""

import pytest
import numpy as np
import anndata as ad
import pandas as pd
from scipy import sparse

from scself._modules.find_modules import get_correlation_modules, get_correlation_submodules

@pytest.fixture
def simple_adata():
    # Create a simple anndata object with clear modular structure
    # Three modules: genes 0-2, 3-5, and 6-8, with genes 9-11 as noise
    X = np.array([
        # Module 1: genes 0-2
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0.2, 0.1, 0.2],
        # Module 2: genes 3-5
        [0, 0, 0, 2, 2, 2, 0, 0, 0, 0.1, 0.1, 0.2],
        [0, 0, 0, 2, 2, 2, 0, 0, 0, 0.2, 0.2, 0.1],
        # Module 3: genes 6-8
        [0, 0, 0, 0, 0, 0, 3, 3, 3, 0.1, 0.1, 0.1],
        [0, 0, 0, 0, 0, 0, 3, 3, 3, 0.2, 0.2, 0.2],
    ])
    return ad.AnnData(
        X,
        var=pd.DataFrame(index=[f'gene_{i}' for i in range(12)]),
        obs=pd.DataFrame(index=[f'cell_{i}' for i in range(6)])
    )

def test_get_correlation_modules_basic(simple_adata):
    # Test basic functionality
    result = get_correlation_modules(simple_adata, n_neighbors=3)
    
    assert 'gene_module' in result.var
    assert 'X_corrcoef' in result.varp
    assert 'X_umap' in result.varm
    assert result.var['gene_module'].dtype.name == 'category'
    
    # Check that we get at least 3 modules (excluding noise)
    assert len(result.var['gene_module'].unique()) >= 3

def test_get_correlation_modules_sparse(simple_adata):
    # Test with sparse input
    simple_adata.X = sparse.csr_matrix(simple_adata.X)
    result = get_correlation_modules(simple_adata, n_neighbors=3)
    
    assert 'gene_module' in result.var
    assert 'X_corrcoef' in result.varp
    assert sparse.issparse(simple_adata.X)

def test_get_correlation_modules_layer(simple_adata):
    # Test with different layer
    simple_adata.layers['test'] = simple_adata.X.copy()
    result = get_correlation_modules(simple_adata, layer='test', n_neighbors=3)
    
    assert 'test_corrcoef' in result.varp
    assert 'test_umap' in result.varm

def test_get_correlation_modules_custom_output(simple_adata):
    # Test custom output key
    result = get_correlation_modules(simple_adata, output_key='custom_modules', n_neighbors=3)
    assert 'custom_modules' in result.var

def test_get_correlation_submodules_basic(simple_adata):
    # Test basic submodule functionality
    simple_adata.var['gene_module'] = [0] * 6 + [1] * 6
    simple_adata.var['gene_module'] = simple_adata.var['gene_module'].astype('category')

    _original = simple_adata.var['gene_module'].copy()
    result = get_correlation_submodules(simple_adata, n_neighbors=2)

    assert all([x == y for x, y in zip(
        [0] * 3 + [1] * 3 + [0] * 3 + [1] * 3,
        result.var['gene_submodule'].tolist()
    )])
    
    assert 'gene_submodule' in result.var
    assert 'X_submodule_umap' in result.varm
    assert result.var['gene_submodule'].dtype.name == 'category'
    
    # Check that original modules are preserved
    assert all(simple_adata.var['gene_module'] == _original)

def test_get_correlation_submodules_missing_input(simple_adata):
    # Test error when input_key is missing
    with pytest.raises(RuntimeError):
        get_correlation_submodules(simple_adata, input_key='nonexistent', n_neighbors=2)

def test_get_correlation_submodules_custom_keys(simple_adata):
    # Test custom input/output keys
    simple_adata.var['custom_modules'] = [0] * 6 + [1] * 6
    simple_adata.var['custom_modules'] = simple_adata.var['custom_modules'].astype('category')

    result = get_correlation_submodules(
        simple_adata,
        input_key='custom_modules',
        output_key='custom_submodules',
        n_neighbors=2
    )

    assert all([x == y for x, y in zip(
        [0] * 3 + [1] * 3 + [0] * 3 + [1] * 3,
        result.var['custom_submodules'].tolist()
    )])
    
    assert 'custom_submodules' in result.var

def test_get_correlation_submodules_layer(simple_adata):
    # Test with different layer
    simple_adata.layers['test'] = simple_adata.X.copy()
    simple_adata.var['gene_module'] = [0] * 6 + [1] * 6
    simple_adata.var['gene_module'] = simple_adata.var['gene_module'].astype('category')

    result = get_correlation_submodules(simple_adata, layer='test', n_neighbors=2)
    
    assert all([x == y for x, y in zip(
        [0] * 3 + [1] * 3 + [0] * 3 + [1] * 3,
        result.var['gene_submodule'].tolist()
    )])

    assert 'test_submodule_umap' in result.varm

def test_get_correlation_submodules_sparse(simple_adata):
    # Test with sparse input
    simple_adata.X = sparse.csr_matrix(simple_adata.X)
    simple_adata.var['gene_module'] = [0] * 6 + [1] * 6
    simple_adata.var['gene_module'] = simple_adata.var['gene_module'].astype('category')

    result = get_correlation_submodules(simple_adata, n_neighbors=2)
    
    assert 'gene_submodule' in result.var
    assert sparse.issparse(simple_adata.X)

    assert all([x == y for x, y in zip(
        [0] * 3 + [1] * 3 + [0] * 3 + [1] * 3,
        result.var['gene_submodule'].tolist()
    )])

def test_module_sizes(simple_adata):
    # Test that modules have reasonable sizes
    result = get_correlation_modules(simple_adata, n_neighbors=2)
    module_sizes = result.var['gene_module'].value_counts()
    
    # Check that the largest modules have at least 3 genes
    assert module_sizes.iloc[0] >= 3
    assert module_sizes.iloc[1] >= 3
    assert module_sizes.iloc[2] >= 3