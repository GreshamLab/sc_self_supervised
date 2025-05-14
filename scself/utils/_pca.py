import scipy.sparse as sps
import anndata as ad
import numpy as np

from sklearn.decomposition import PCA

def pca(X, n_pcs, zero_center=True):

    _is_adata = isinstance(X, ad.AnnData)

    if _is_adata:
        X_ref = X.X
    else:
        X_ref = X

    if zero_center:
        import scanpy as sc
        from scself.utils import sparse_dot_patch

        if sps.issparse(X_ref):
            sparse_dot_patch(X_ref)

        return sc.pp.pca(
            X,
            n_comps=n_pcs
        )

    else:
        try:
            from scself.sparse.truncated_svd import TruncatedSVDMKL as TruncatedSVD
        except ImportError:
            from sklearn.decomposition import TruncatedSVD

        scaler = TruncatedSVD(n_components=n_pcs)
        _pca_data = scaler.fit_transform(X_ref)

        if _is_adata:

            X.obsm['X_pca'] = _pca_data
            X.varm['PCs'] = scaler.components_.T
            X.uns['pca'] = {
                "variance": scaler.explained_variance_,
                "variance_ratio": scaler.explained_variance_ratio_,
            }

            return X

        else:
            return _pca_data


def stratified_pca(
    adata,
    obs_col,
    n_comps=50,
    random_state=100,
    n_per_group=None,
    layer='X'
):
    """Perform PCA on a stratified subset of cells and project results to full dataset.

    Takes a balanced random sample from each group defined by obs_col to compute PCA,
    then projects the loadings onto the full dataset. This helps prevent the PCA from
    being dominated by more abundant cell types.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix
    obs_col : str
        Column in adata.obs containing group labels to stratify by
    random_state : int, optional
        Random seed for reproducibility, by default 100
    n_per_group : int, optional
        Number of cells to sample per group. If None, uses size of smallest group
    layer : str, optional
        Layer in adata.layers to use. If 'X', uses adata.X, by default 'X'

    Returns
    -------
    anndata.AnnData
        Input adata with new obsm['{layer}_pca_stratified'] containing PCA coordinates
    """
    # Initialize random number generator
    rng = np.random.default_rng(random_state)
    
    # Get counts of cells in each group
    group_counts = adata.obs[obs_col].value_counts()

    # If n_per_group not specified, use size of smallest group
    if n_per_group is None:
        n_per_group = min(group_counts)

    # Sample cells from each group
    keep_idx = []
    for ct, x in group_counts.items():
        keep_idx.extend(
            rng.choice(
                np.where(adata.obs[obs_col] == ct)[0],
                size=min(x, n_per_group),
                replace=False
            )
        )
    
    # Get expression matrix from specified layer
    lref = adata.X if layer == 'X' else adata.layers[layer]

    # Fit PCA to balanced sample
    pca_ = PCA(
        n_components=n_comps,
        svd_solver='arpack',
        random_state=random_state,
    ).fit(lref[keep_idx, :])

    # Project PCA onto full dataset and pack anndata object
    adata.obsm[f'{layer}_pca_stratified'] = pca_.transform(lref)
    adata.varm[f'{layer}_stratified_PCs'] = pca_.components_.T

    adata.uns['pca_stratified'] = dict(
        variance=pca_.explained_variance_,
        variance_ratio=pca_.explained_variance_ratio_,
    )

    return adata
