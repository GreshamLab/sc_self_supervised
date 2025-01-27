import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.neighbors import KNeighborsTransformer, kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scself.utils import corrcoef


### Calculate k-NN from a distance matrix directly in scanpy
class KNeighborsTransformerPassthrough(KNeighborsTransformer):

    def fit_transform(self, X):
        return kneighbors_graph(
            X,
            metric='precomputed',
            n_neighbors=self.n_neighbors
        )


def get_correlation_modules(
    adata,
    layer='X',
    n_neighbors=10,
    leiden_kwargs={},
    reorder_genes=True,
    output_key='gene_module'
):
    """
    Get correlation modules from an anndata object.

    Adds .varp['X_corrcoef'] with gene-gene correlation
    and .var[output_key] with gene module ID

    :param adata: Data object containing expression data
    :type adata: ad.AnnData
    :param layer: Layer to calculate correlation and find
        modules from, defaults to 'X'
    :type layer: str, optional
    :param n_neighbors: Number of neighbors in kNN, defaults
        to 10
    :type n_neighbors: int, optional
    :param leiden_kwargs: Keyword arguments to sc.tl.leiden
    :type leiden_kwargs: dict, optional
    :param reorder_genes: Reorder returned object for plotting,
        by heirarchial clustering, defaults to True
    :type reorder_genes: bool, optional    
    :return: A [genes x genes] AnnData object with
        correlation distance in .X and gene modules in
        .obs[output_key].
    :rtype: ad.AnnData
    """

    lref = adata.X if layer == 'X' else adata.layers[layer]

    adata.varp[f'{layer}_corrcoef'] = corrcoef(lref)

    corrs_dists = adata.varp[f'{layer}_corrcoef'] * -1
    corrs_dists += 1

    if reorder_genes:
        # Order by hclust on correlation distance
        corr_order = np.array(
            dendrogram(
                linkage(
                    squareform(corrs_dists, checks=False),
                    method="average"
                ),
                no_plot=True
            )["leaves"]
        )

        corr_dist_adata = ad.AnnData(
            corrs_dists[corr_order, :][:, corr_order],
            var=pd.DataFrame(index=adata.var_names[corr_order]),
            obs=pd.DataFrame(index=adata.var_names[corr_order])
        )

    else:
        corr_dist_adata = ad.AnnData(
            corrs_dists,
            var=pd.DataFrame(index=adata.var_names),
            obs=pd.DataFrame(index=adata.var_names)
        )

    corr_dist_adata.var = corr_dist_adata.var.join(
        adata.var
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

    corr_dist_adata.obs['leiden'] = corr_dist_adata.obs['leiden'].astype(
        int
    ).astype(
        'category'
    )

    corr_dist_adata.obs[output_key] = corr_dist_adata.obs['leiden']
    corr_dist_adata.var[output_key] = corr_dist_adata.obs['leiden']

    adata.var = adata.var.join(corr_dist_adata.var[[output_key]])

    return corr_dist_adata


def module_score(
    adata,
    genes,
    layer='X',
    scaler=StandardScaler(),
    fit_scaler=True,
    clipping=(-10, 10),
    **kwargs
):
    """
    Calculate a module score from a set of genes
    by zscoring each gene, clipping to [-10, 10], and averaging
    the gene zscores for each observation.

    Casts to dense array unless the data is sparse CSR.

    :param adata: AnnData with gene expression
    :type adata: ad.AnnData
    :param genes: List of genes to use for the module
    :type genes: list, tuple, pd.Index, pd.Series
    :param layer: Data layer to use for scoring, defaults to 'X'
    :type layer: str, optional
    :param scaler: Scaling transformer from sklearn,
        defaults to StandardScaler()
    :type scaler: sklearn.Transformer, optional
    :param fit_scaler: Fit scaler to the data (fit_transform),
        instead of just using it (transform), defaults to True
    :type fit_scaler: bool, optional
    :param clipping: Clip post-scaled results to a range,
        defaults to (-10, 10)
    :type clipping: tuple, optional
    :return: Score for every observation in the array
    :rtype: np.ndarray
    """

    if layer == 'X':
        dref = adata.X
    else:
        dref = adata.layers[layer]

    _data = dref[:, adata.var_names.isin(genes)]

    try:
        _data = _data.toarray()
    except AttributeError:
        _data = _data.copy()

    if scaler is None:
        _scores = _data
    else:
        # Allow for uninstantiated scalers to
        # be passed in
        try:
            scaler = scaler(**kwargs)
        except TypeError:
            pass

        # Either fit the scaler and then use it
        # or use it without fitting, depending on flag
        if fit_scaler:
            _scores = scaler.fit_transform(_data)
        else:
            _scores = scaler.transform(_data)

    if clipping is not None:
        np.clip(_scores, *clipping, out=_scores)

    return np.mean(
        _scores,
        axis=1
    )
