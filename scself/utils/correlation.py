import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
import scipy.sparse as sps

from sklearn.neighbors import KNeighborsTransformer, kneighbors_graph

from scself.utils.dot_product import dot

### COV AND CORRCOEF ###
### LIGHER WEIGHT THAN NUMPY ###

### Calculate k-NN from a distance matrix directly in scanpy
class KNeighborsTransformerPassthrough(KNeighborsTransformer):

    def fit_transform(self, X):
        return kneighbors_graph(
            X,
            metric='precomputed',
            n_neighbors=self.n_neighbors
        )


def cov(X, axis=0):

    if sps.issparse(X):
        return cov_sparse(X, axis=axis)

    # Center and get num rows
    avg, w_sum = np.average(X, axis=axis, weights=None, returned=True)
    w_sum = w_sum[0]
    X = X - (avg[None, :] if axis == 0 else avg[:, None])

    # Gram matrix
    if axis == 0:
        X = np.dot(X.T, X)
    else:
        X = np.dot(X, X.T)

    X *= np.true_divide(1, w_sum - 1)

    return X

def cov_sparse(X, axis=0):

    axsum = X.sum(axis)
    w_sum = X.shape[axis]

    # for spmatrix & sparray
    try:
        axsum = axsum.A1
    except AttributeError:
        axsum = axsum.ravel()

    axsum = axsum.reshape(-1, 1).dot(axsum.reshape(1, -1)) 
    axsum /= w_sum

    if axis == 0:
        X_cov = dot(X.T, X, dense=True)
    else:
        X_cov = dot(X, X.T, dense=True)

    X_cov -= axsum
    X_cov /= (w_sum - 1)

    return X_cov

def corrcoef(X, axis=0):

    X = cov(X, axis=axis)
    sd = np.sqrt(np.diag(X))

    _zero_var = sd == 0
    sd[_zero_var] = 1.

    X /= sd[:, None]
    X /= sd[None, :]

    # Fixes for float precision
    np.clip(X, -1, 1, out=X)
    np.fill_diagonal(X, 1.)

    return X


def correlation_clustering_and_umap(
    correlations,
    n_neighbors=10,
    var_names=None,
    **leiden_kwargs
):
    
    corr_dist_adata = ad.AnnData(
        1 - correlations,
        var=pd.DataFrame(index=var_names) if var_names is not None else None,
        obs=pd.DataFrame(index=var_names) if var_names is not None else None
    )

    # Special case handling when there are too many neighbors for
    # the provided data; comes up with submodules a lot
    if corr_dist_adata.shape[0] <= n_neighbors:

        n_neighbors = corr_dist_adata.shape[0] - 2

        if n_neighbors <= 1:
            corr_dist_adata.obs['leiden'] = '0'
            corr_dist_adata.obs['leiden'] = corr_dist_adata.obs['leiden'].astype('category')
            corr_dist_adata.obsm['X_umap'] = np.zeros((corr_dist_adata.shape[0], 2), float)
            return corr_dist_adata
    
        skip_leiden = True
    else:
        skip_leiden = False

    # Build kNN and get modules by graph clustering
    sc.pp.neighbors(
        corr_dist_adata,
        n_neighbors=n_neighbors,
        transformer=KNeighborsTransformerPassthrough(
            n_neighbors=n_neighbors
        ),
        use_rep='X'
    )
    sc.tl.umap(corr_dist_adata)

    if skip_leiden:
        corr_dist_adata.obs['leiden'] = '0'
        corr_dist_adata.obs['leiden'] = corr_dist_adata.obs['leiden'].astype('category')
    else:
        sc.tl.leiden(corr_dist_adata, **leiden_kwargs)

    return corr_dist_adata
