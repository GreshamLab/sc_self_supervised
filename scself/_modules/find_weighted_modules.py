import pandas as pd
import numpy as np
import anndata as ad

from functools import reduce
from scipy.sparse import csr_matrix

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
    Find gene modules by combining correlation patterns across multiple datasets.

    This function performs a weighted averaging of gene-gene correlations across
    multiple AnnData objects, enabling module discovery that accounts for genes
    measured in different subsets of datasets. The workflow consists of:

    1. Calculate gene-gene correlation within each dataset independently
    2. Determine the union of genes across all datasets
    3. Average correlations where genes overlap (weighted by dataset availability)
    4. Cluster averaged correlations using Leiden community detection
    5. Assign module IDs back to individual datasets

    The key advantage is handling datasets with non-overlapping gene sets while
    properly weighting correlations based on the number of datasets in which both
    genes appear.

    **Modifies input datasets in-place:**
        - Adds .varp['{layer}_corrcoef'] with gene-gene correlation matrix
        - Adds .var[output_key] with integer gene module assignments
        - Adds .varm['{layer}_umap'] with UMAP coordinates in shared space

    :param adata_list: List of AnnData objects containing expression data.
        Datasets may have different genes but should have comparable expression scales.
    :type adata_list: list(ad.AnnData)
    :param layer: Layer to use for correlation calculation. Can be 'X' or any layer name.
        Accepts either a single string (applied to all datasets) or a list with one
        layer name per dataset, defaults to 'X'
    :type layer: str or list(str), optional
    :param n_neighbors: Number of neighbors for kNN graph construction used in
        clustering and UMAP, defaults to 10
    :type n_neighbors: int, optional
    :param leiden_kwargs: Additional keyword arguments passed to leiden clustering
        (e.g., resolution parameter), defaults to {}
    :type leiden_kwargs: dict, optional
    :param output_key: Column name to add to .var for storing module IDs,
        defaults to 'gene_module'
    :type output_key: str, optional
    :param obs_mask: Boolean mask or slice to subset observations (cells) before
        computing correlations. Can be a single mask (applied to all) or list of masks
        (one per dataset). Useful for computing modules from specific cell populations.
    :type obs_mask: np.ndarray, slice, list, or None, optional

    :return: An AnnData object representing the full gene-gene correlation matrix with:
        - .layers['{layer}_corrcoef']: Averaged correlation matrix (n_genes x n_genes)
        - .obs['leiden'] and .var['leiden']: Module assignments (symmetric)
        - .obsm['{layer}_umap']: UMAP embedding of gene correlations
    :rtype: ad.AnnData
    """

    _n_datasets = len(adata_list)

    # Initialize obs_mask to None for all datasets if not provided
    if obs_mask is None:
        obs_mask = [None] * _n_datasets

    # Helper function to broadcast scalar arguments to lists
    # matching the number of datasets. This allows users to pass
    # a single layer name that applies to all datasets, or a list
    # with one layer name per dataset.
    def _to_iterable(arg, argname):
        if not isinstance(arg, _ITER_TYPES):
            # Broadcast scalar to list
            return [arg] * _n_datasets
        elif len(arg) != _n_datasets:
            raise AttributeError(
                f"len({argname}) = {len(arg)}; {_n_datasets} is required"
            )
        else:
            return arg

    layer = _to_iterable(layer, 'layer')

    # Step 1: Calculate gene-gene correlation for each dataset independently
    # Store results in each dataset's .varp to enable caching across calls
    for adata, layer_i, mask_i in zip(
        adata_list,
        layer,
        obs_mask
    ):
        # Skip if correlation already computed for this layer
        if f'{layer_i}_corrcoef' not in adata.varp.keys():
            # Get reference to expression data (X or specified layer)
            _lref = adata.X if layer_i == 'X' else adata.layers[layer_i]

            # Apply observation mask if provided (subset to specific cells)
            # Then compute gene-gene correlation matrix
            adata.varp[f'{layer_i}_corrcoef'] = corrcoef(
                _lref[mask_i, :] if mask_i is not None else _lref
            )

            del _lref

    # Step 2: Determine the gene universe across all datasets
    # Check if all datasets share identical gene names (same genes, same order)
    if all(
        all(
            adata.var_names.equals(a.var_names)
            for a in adata_list
        )
        for adata in adata_list
    ):
        # All datasets aligned - no reindexing needed
        _genes = adata_list[0].var_names.copy()
        _do_reindex=False
    else:
        # Datasets have different genes - compute union of all gene names
        _genes = reduce(
            lambda x, y: x.var_names.union(y.var_names),
            adata_list
        )
        _do_reindex=True

    _n_genes = len(_genes)

    # Count how many datasets contain each gene
    # This is used later for weighted averaging of correlations
    _gene_counts = reduce(
        lambda x, y: x + y,
        [_genes.isin(c.var_names).astype(int) for c in adata_list]
    )

    # Step 3: Initialize combined correlation matrix
    # Create an AnnData object to hold the gene x gene correlation
    # Both obs and var represent genes (symmetric gene-gene matrix)
    full_correlation = ad.AnnData(
        csr_matrix((_n_genes, _n_genes)),
        var=pd.DataFrame(index=_genes),
        obs=pd.DataFrame(index=_genes)
    )

    # Store the summed correlations in a layer (will average later)
    _corr_layer = f'{layer[0]}_corrcoef'
    full_correlation.layers[_corr_layer] = np.zeros(
        (_n_genes, _n_genes),
        dtype=float
    )

    # Step 4: Accumulate correlation matrices from each dataset
    # Sum correlations into the appropriate positions based on gene names
    for adata, layer_i in zip(
        adata_list,
        layer
    ):

        if _do_reindex:
            # Map each dataset's genes to their positions in the full gene universe
            # np.ix_ creates a mesh for 2D indexing (row indices x column indices)
            full_correlation.layers[_corr_layer][
                np.ix_(
                    full_correlation.obs_names.get_indexer(adata.var_names),
                    full_correlation.var_names.get_indexer(adata.var_names)
                )
            ] += adata.varp[f'{layer_i}_corrcoef']
        else:
            # All datasets aligned - direct addition
            full_correlation.layers[_corr_layer] += adata.varp[f'{layer_i}_corrcoef']

    # Step 5: Compute weighted average of correlations
    # Divide by the minimum count between gene pairs to get proper average
    # Example: if gene A appears in 3 datasets and gene B in 2 datasets,
    # their correlation can only be calculated in min(3,2)=2 datasets
    full_correlation.layers[_corr_layer] /= np.minimum(
        _gene_counts[:, None],  # Row-wise gene counts
        _gene_counts[None, :]   # Column-wise gene counts
    )

    # Sanity check: correlations should be in [-1, 1]
    assert full_correlation.layers[_corr_layer].max() <= 1.0

    # Step 6: Cluster genes based on correlation patterns
    # Build kNN graph, run Leiden clustering, and compute UMAP embedding
    _corr_results = correlation_clustering_and_umap(
        full_correlation.layers[_corr_layer],
        n_neighbors=n_neighbors,
        var_names=full_correlation.var_names,
        **leiden_kwargs
    )

    # Store clustering results in both obs and var (symmetric since genes x genes)
    full_correlation.obs['leiden'] = _corr_results.obs['leiden'].astype(int).values
    full_correlation.var['leiden'] = _corr_results.obs['leiden'].astype(int).values
    full_correlation.obsm[f'{layer_i}_umap'] = _corr_results.obsm['X_umap']

    # Step 7: Propagate module assignments back to individual datasets
    # Each dataset gets the module IDs for its own genes
    for adata, layer_i in zip(
        adata_list,
        layer
    ):
        # Find positions of this dataset's genes in the full correlation results
        _gene_idx = _corr_results.var_names.get_indexer(adata.var_names)

        # Assign module membership to each gene
        adata.var[output_key] = _corr_results.obs['leiden']

        # Store the UMAP coordinates for this dataset's genes
        # This allows plotting genes from different datasets in the same UMAP space
        adata.varm[f'{layer_i}_umap'] = _corr_results.obsm['X_umap'][_gene_idx, :]

    return full_correlation
