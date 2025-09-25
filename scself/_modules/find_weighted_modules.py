import pandas as pd
import numpy as np
import anndata as ad

from functools import reduce

from scself.utils.correlation import (
    corrcoef,
    correlation_clustering_and_umap
)

_ITER_TYPES = (tuple, list, pd.Series, np.ndarray)

def get_combined_correlation_modules(
    adata_list,
    layer='X',
    n_neighbors=10,
    leiden_kwargs={},
    output_key='gene_module',
    obs_mask=None
):
    """
    Get correlation modules from a list of anndata objects
    by calculating the correlation separately for each object
    and averaging the correlation between genes where the
    objects overlap.

    Adds .varp['X_corrcoef'] with gene-gene correlation
    and .var[output_key] with gene module ID

    :param adata_list: List of data objects containing
        expression data
    :type adata_list: list(ad.AnnData)
    :param layer: Layer to calculate correlation and find
        modules from, can provide a list with the same length
        as adata_list, defaults to 'X'
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

    :return: A correlation (gene x gene) anndata object with:
        Gene-gene correlation UMAP in 'X_umap' in .varm
        Module membership IDs in .obs[output_key]
    :rtype: ad.AnnData
    """

    _n_datasets = len(adata_list)

    if obs_mask is None:
        obs_mask = [None] * _n_datasets

    # Make sure all of the arguments are iterable lists of
    # correct length and raise an AttributeError if not
    def _to_iterable(arg, argname):
        if not isinstance(arg, _ITER_TYPES):
            return [arg] * _n_datasets
        elif len(arg) != _n_datasets:
            raise AttributeError(
                f"len({argname}) = {len(arg)}; {_n_datasets} is required"
            )
        else:
            return arg

    layer = _to_iterable(layer, 'layer')

    # Calculate correlation for each dataset
    for adata, layer_i, mask_i in zip(
        adata_list,
        layer,
        obs_mask
    ):
        
        if f'{layer_i}_corrcoef' not in adata.varp.keys():
            _lref = adata.X if layer_i == 'X' else adata.layers[layer_i]

            adata.varp[f'{layer_i}_corrcoef'] = corrcoef(
                _lref[mask_i, :] if mask_i is not None else _lref
            )

            del _lref

    # Check to see if all the data is already aligned
    # If not, find the union of all the var_names
    if all(
        all(
            adata.var_names.equals(a.var_names)
            for a in adata_list
        )
        for adata in adata_list
    ):
        _genes = adata_list[0].var_names.copy()
        _do_reindex=False
    else:
        _genes = reduce(
            lambda x, y: x.var_names.union(y.var_names),
            adata_list
        )
        _do_reindex=True

    _n_genes = len(_genes)

    # Get the number of times each gene appears
    _gene_counts = reduce(
        lambda x, y: x + y,
        [_genes.isin(c.var_names).astype(int) for c in adata_list]
    )

    # Create a zeroed correlation matrix for the
    # gene union
    full_correlation = ad.AnnData(
        np.zeros(
            (_n_genes, _n_genes),
            dtype=float
        ),
        var=pd.DataFrame(index=_genes),
        obs=pd.DataFrame(index=_genes)
    )

    # Iterate through all the anndata object and add
    # each correlation into the full_correlation
    # indexed appropriately for the feaure name
    for adata, layer_i in zip(
        adata_list,
        layer
    ):
        
        if _do_reindex:
            full_correlation.X[
                np.ix_(
                    full_correlation.obs_names.get_indexer(adata.var_names),
                    full_correlation.var_names.get_indexer(adata.var_names)
                )
            ] += adata.varp[f'{layer_i}_corrcoef']
        else:
            full_correlation.X += adata.varp[f'{layer_i}_corrcoef']
    
    # Correct for the number of times the gene-gene correlation
    # was calculated
    full_correlation.X /= np.minimum(
        _gene_counts[:, None],
        _gene_counts[None, :]
    )
    
    full_correlation = correlation_clustering_and_umap(
        full_correlation.X,
        n_neighbors=n_neighbors,
        var_names=full_correlation.var_names,
        **leiden_kwargs
    )

    full_correlation.obs['leiden'] = full_correlation.obs['leiden'].astype(int)

    for adata, layer_i in zip(
        adata_list,
        layer
    ):
        _gene_idx = full_correlation.var_names.get_indexer(adata.var_names)

        # Put the gene module memberships in
        adata.var[output_key] = full_correlation.obs['leiden']

        # Put the partial umap into the separate objects
        # so they can be plotted in the same space
        adata.varm[f'{layer_i}_umap'] = full_correlation.obsm['X_umap'][_gene_idx, :]

    return full_correlation
