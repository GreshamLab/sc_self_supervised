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
        return cov_sparse(X)

    # Center and get num rows
    avg, w_sum = np.average(X, axis=axis, weights=None, returned=True)
    w_sum = w_sum[0]
    X = X - (avg[None, :] if axis == 0 else avg[:, None])

    # Gram matrix
    X = np.dot(X.T, X)
    X *= np.true_divide(1, w_sum)

    return X

def cov_sparse(X, axis=0):

    avg = X.mean(axis)

    # for spmatrix & sparray
    try:
        avg = avg.A1
    except AttributeError:
        avg = avg.ravel()

    w_sum = X.shape[axis]
    X = dot(X.T, X, dense=True)
    X *= np.true_divide(1, w_sum)

    return X

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
        var=pd.DataFrame(index=var_names),
        obs=pd.DataFrame(index=var_names)
    )

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
    sc.tl.leiden(corr_dist_adata, **leiden_kwargs)

    return corr_dist_adata
