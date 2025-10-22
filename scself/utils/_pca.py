"""
Principal Component Analysis (PCA) utilities for single-cell data.

This module provides optimized PCA functions that leverage Intel MKL for
sparse matrix operations when available, and includes stratified PCA for
handling imbalanced cell type compositions.
"""

import scipy.sparse as sps
import anndata as ad
import numpy as np

from sklearn.decomposition import PCA
from scself.sparse import is_csr


def pca(
    X,
    n_pcs,
    layer=None,
    **kwargs
):
    """
    Compute PCA with optional Intel MKL acceleration for sparse matrices.

    This function wraps scanpy's PCA implementation and automatically uses
    Intel MKL sparse matrix operations (via sparse_dot_mkl) when available
    for significant performance improvements. Falls back to standard scanpy
    PCA if MKL is not installed.

    The function temporarily converts scipy CSR matrices to MKL-backed CSR
    matrices for computation, then converts back to maintain compatibility.

    :param X: Input data. Can be AnnData object or raw matrix.
    :type X: ad.AnnData or np.ndarray or scipy.sparse matrix
    :param n_pcs: Number of principal components to compute
    :type n_pcs: int
    :param layer: Name of layer to use if X is AnnData. None or 'X' uses
        the main data matrix. Defaults to None.
    :type layer: str or None, optional
    :param kwargs: Additional keyword arguments passed to scanpy.pp.pca
    :return: If X is AnnData, returns modified X with PCA results added.
        Otherwise returns PCA result.
    :rtype: ad.AnnData or array-like
    """

    import scanpy as sc

    # Check if input is AnnData and which layer to use
    _is_adata = isinstance(X, ad.AnnData)

    # Accept layer = 'X' for the main data .X
    if layer == 'X':
        layer = None

    _layer_X = layer is None

    # Try to import Intel MKL sparse matrix library
    try:
        from sparse_dot_mkl import csr_matrix as mkl_csr_matrix

    except ImportError:
        # MKL not available, use standard scanpy PCA
        return sc.pp.pca(
            X,
            n_comps=n_pcs,
            layer=layer,
            **kwargs
        )

    # Convert CSR matrices to MKL-backed format for acceleration
    _revert = False
    if _is_adata and _layer_X:
        # Convert main data matrix if it's CSR
        if is_csr(X.X):
            X.X = mkl_csr_matrix(X.X)
            _revert = True
    elif _is_adata:
        # Convert specified layer if it's CSR
        if is_csr(X.layers[layer]):
            X.layers[layer] = mkl_csr_matrix(X.layers[layer])
            _revert = True
    elif is_csr(X):
        # Convert raw matrix if it's CSR
            X = mkl_csr_matrix(X)

    try:
        # Compute PCA with MKL acceleration
        return sc.pp.pca(
            X,
            n_comps=n_pcs,
            layer=layer,
            **kwargs
        )
    finally:
        # Revert MKL matrices back to standard scipy format
        if _revert and _layer_X:
            X.X = sps.csr_matrix(X.X)
        elif _revert:
            X.layers[layer] = sps.csr_matrix(X.layers[layer])


def stratified_pca(
    adata,
    obs_col,
    n_comps=50,
    random_state=100,
    n_per_group=None,
    layer='X'
):
    """
    Perform PCA on a stratified subset of cells and project to full dataset.

    This function computes PCA on a balanced random sample from each group
    (e.g., cell type) rather than the full dataset. This prevents the PCA
    from being dominated by abundant cell types and ensures all groups
    contribute equally to the principal components. The learned PCA loadings
    are then projected onto the entire dataset.

    This is particularly useful for:
    - Datasets with highly imbalanced cell type compositions
    - Visualizations where rare cell types need better representation
    - Analyses requiring unbiased dimensionality reduction

    The function stores PCA results in adata following scanpy conventions:
    - obsm['{layer}_pca_stratified']: PCA coordinates for all cells
    - varm['{layer}_stratified_PCs']: Principal component loadings
    - uns['pca_stratified']: Variance and variance ratio explained

    :param adata: Annotated data matrix containing expression data
    :type adata: anndata.AnnData
    :param obs_col: Column name in adata.obs containing group labels for
        stratification. Each unique value defines a group from which cells
        will be sampled.
    :type obs_col: str
    :param n_comps: Number of principal components to compute. Defaults to 50.
    :type n_comps: int, optional
    :param random_state: Random seed for reproducible sampling. Defaults to 100.
    :type random_state: int, optional
    :param n_per_group: Number of cells to sample from each group. If None,
        uses the size of the smallest group (ensures perfect balance). If
        specified and larger than a group's size, samples all cells from that
        group. Defaults to None.
    :type n_per_group: int or None, optional
    :param layer: Layer name in adata.layers to use for PCA. If 'X', uses
        adata.X (main data matrix). Defaults to 'X'.
    :type layer: str, optional
    :return: Input adata object with PCA results added in obsm, varm, and uns
    :rtype: anndata.AnnData

    Examples:
        >>> # Compute PCA using balanced samples from each cell type
        >>> stratified_pca(adata, obs_col='cell_type', n_comps=50)
        >>>
        >>> # Sample 100 cells per cell type
        >>> stratified_pca(adata, obs_col='cell_type', n_per_group=100)
        >>>
        >>> # Use a specific layer
        >>> stratified_pca(adata, obs_col='batch', layer='log_normalized')
    """
    # Initialize random number generator for reproducible sampling
    rng = np.random.default_rng(random_state)

    # Get counts of cells in each group
    group_counts = adata.obs[obs_col].value_counts()

    # Determine sample size per group
    if n_per_group is None:
        # Use smallest group size to ensure perfect balance
        n_per_group = min(group_counts)

    # Collect indices of cells to use for PCA fitting
    keep_idx = []
    for ct, x in group_counts.items():
        # Sample cells from this group (or take all if group is smaller than n_per_group)
        keep_idx.extend(
            rng.choice(
                np.where(adata.obs[obs_col] == ct)[0],
                size=min(x, n_per_group),
                replace=False
            )
        )

    # Get expression matrix from specified layer
    lref = adata.X if layer == 'X' else adata.layers[layer]

    # Fit PCA to the balanced sample using ARPACK solver
    # (efficient for sparse matrices and moderate n_components)
    pca_ = PCA(
        n_components=n_comps,
        svd_solver='arpack',
        random_state=random_state,
    ).fit(lref[keep_idx, :])

    # Project learned PCA onto entire dataset
    adata.obsm[f'{layer}_pca_stratified'] = pca_.transform(lref)

    # Store principal component loadings (genes x components)
    adata.varm[f'{layer}_stratified_PCs'] = pca_.components_.T

    # Store variance explained statistics
    adata.uns['pca_stratified'] = dict(
        variance=pca_.explained_variance_,
        variance_ratio=pca_.explained_variance_ratio_,
    )

    return adata
