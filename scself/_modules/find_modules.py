import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc

from sklearn.neighbors import KNeighborsTransformer, kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scself.utils.correlation import (
    corrcoef,
    correlation_clustering_and_umap
)

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
    :param output_key: Column to add to adata.var with module IDs,
        defaults to 'gene_module'
    :type output_key: str, optional

    :return: The original adata object with:
        Gene correlations in 'X_corrcoef' in .varp
        Gene-gene correlation UMAP in 'X_umap' in .varm
        Module membership IDs in .var[output_key]
    :rtype: ad.AnnData
    """

    if f'{layer}_corrcoef' not in adata.varp.keys():
        adata.varp[f'{layer}_corrcoef'] = corrcoef(
            adata.X if layer == 'X' else adata.layers[layer]
        )

    corr_dist_adata = correlation_clustering_and_umap(
        adata.varp[f'{layer}_corrcoef'],
        n_neighbors=n_neighbors,
        var_names=adata.var_names,
        **leiden_kwargs
    )

    adata.var[output_key] = corr_dist_adata.obs['leiden'].astype(
        int
    ).astype(
        'category'
    )

    adata.varm[f'{layer}_umap'] = corr_dist_adata.obsm['X_umap']

    return adata


def get_correlation_submodules(
    adata,
    layer='X',
    n_neighbors=10,
    leiden_kwargs={},
    input_key='gene_module',
    output_key='gene_submodule'
):
    """
    Get correlation submodules iteratively from an anndata object
    that contains count data and correlation modules

    :param adata: Data object containing expression data and
        correlation modules in .var
    :type adata: ad.AnnData
    :param layer: Layer to calculate correlation and find
        submodules from, defaults to 'X'
    :type layer: str, optional
    :param n_neighbors: Number of neighbors in kNN, defaults
        to 10
    :type n_neighbors: int, optional
    :param leiden_kwargs: Keyword arguments to sc.tl.leiden
    :type leiden_kwargs: dict, optional
    :param input_key: Column in .var with module IDs,
        defaults to 'gene_module'
    :type input_key: str, optional
    :param output_key: Column to add to adata.var with module IDs,
        defaults to 'gene_submodule'
    :type output_key: str, optional

    :return: The original adata object with:
        Gene-gene submodule correlation UMAP in 'X_submodule_umap' in .varm
        Module submembership IDs in .var[output_key]
    :rtype: ad.AnnData
    """

    if input_key not in adata.var.columns:
        raise RuntimeError(f"Column {input_key} not present in .var")

    lref = adata.X if layer == 'X' else adata.layers[layer]

    adata.var[output_key] = -1
    adata.varm[f'{layer}_submodule_umap'] = np.zeros(
        (adata.shape[1], 2),
        float
    )

    for cat in adata.var[input_key].cat.categories:

        if cat == -1:
            continue

        _slice_idx = adata.var[input_key] == cat

        _slice_corr_dist_adata = correlation_clustering_and_umap(
            corrcoef(lref[:, _slice_idx]),
            n_neighbors=n_neighbors,
            var_names=adata.var_names[_slice_idx],
            **leiden_kwargs
        )
        _slice_corr_dist_adata.obs['leiden'] = _slice_corr_dist_adata.obs['leiden'].astype(int)

        adata.var.loc[
            _slice_corr_dist_adata.obs.index,
            output_key
        ] = _slice_corr_dist_adata.obs['leiden']

        adata.varm[f'{layer}_submodule_umap'][
            adata.var.index.get_indexer(
                _slice_corr_dist_adata.obs.index
            ),
            :
        ] = _slice_corr_dist_adata.obsm['X_umap']

        del _slice_corr_dist_adata

    adata.var[output_key] = adata.var[output_key].astype('category')

    return adata


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
