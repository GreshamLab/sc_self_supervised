import scipy.sparse as sps
import anndata as ad


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
