import numpy as np

from scself.utils.correlation import (
    corrcoef,
    correlation_clustering_and_umap
)

def get_correlation_modules(
    adata,
    layer='X',
    n_neighbors=10,
    leiden_kwargs={},
    output_key='gene_module',
    obs_mask=None
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
    :param obs_mask: Boolean mask or slice for observations to consider
    :type obs_mask: np.ndarray or slice, optional

    :return: The original adata object with:
        Gene correlations in 'X_corrcoef' in .varp
        Gene-gene correlation UMAP in 'X_umap' in .varm
        Module membership IDs in .var[output_key]
    :rtype: ad.AnnData
    """

    if obs_mask is None:
        obs_mask = slice(None)

    if f'{layer}_corrcoef' not in adata.varp.keys():
        adata.varp[f'{layer}_corrcoef'] = corrcoef(
            adata.X[obs_mask, :] if layer == 'X' else
            adata.layers[layer][obs_mask, :]
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
    output_key='gene_submodule',
    obs_mask=None
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
    :param obs_mask: Boolean mask or slice for observations to consider
    :type obs_mask: np.ndarray or slice, optional

    :return: The original adata object with:
        Gene-gene submodule correlation UMAP in 'X_submodule_umap' in .varm
        Module submembership IDs in .var[output_key]
    :rtype: ad.AnnData
    """

    if input_key not in adata.var.columns:
        raise RuntimeError(f"Column {input_key} not present in .var")
    
    if obs_mask is None:
        obs_mask = slice(None)

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
        _slice_idx = _slice_idx.values

        _slice_corr_dist_adata = correlation_clustering_and_umap(
            corrcoef(lref[:, _slice_idx][obs_mask, :]),
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
