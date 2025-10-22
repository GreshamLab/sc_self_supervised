"""
Standardization utilities for single-cell RNA-seq data.

This module provides functions for normalizing, scaling, and transforming count data
from single-cell experiments. Supports depth normalization, log transformation, and
robust scaling with both dense and sparse (CSR) matrix formats.
"""

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sps

from scself.sparse import is_csr, is_csc
from scself.scaling import TruncRobustScaler
from scself.utils.sum import array_sum
from scself.utils.cast_dtype_inplace import cast_to_float_inplace


def _normalize(
    count_data,
    target_sum=None,
    log=False,
    scale=False,
    scale_factor=None,
    size_factor=None,
    subset_genes_for_depth=None,
    stratification_column=None,
    size_factor_cap=None,
    depth_by_sampling=False,
    layer='X',
    random_state=100
):
    """
    Normalize count data with optional log transformation and scaling.

    This is the internal normalization function that performs depth normalization,
    log transformation, and/or robust scaling on count data. All operations are
    performed in-place to minimize memory usage.

    :param count_data: AnnData object containing count data
    :type count_data: ad.AnnData
    :param target_sum: Target sum for depth normalization. If None, uses median
        of total counts per cell. Ignored if size_factor is provided.
    :type target_sum: numeric, optional
    :param log: Whether to apply log1p transformation after normalization
    :type log: bool, optional
    :param scale: Whether to apply robust scaling (per-gene) after normalization
    :type scale: bool, optional
    :param scale_factor: Pre-computed gene-wise scale factors for scaling. If None,
        scale factors are computed using TruncRobustScaler
    :type scale_factor: np.ndarray, optional
    :param size_factor: Pre-computed cell-wise size factors for normalization. If
        provided, target_sum, stratification_column, and size_factor_cap are ignored
    :type size_factor: np.ndarray, optional
    :param subset_genes_for_depth: Boolean mask or indices to select genes for
        computing depth (total counts). Useful for excluding spike-ins or other
        features when calculating normalization factors
    :type subset_genes_for_depth: array-like, optional
    :param stratification_column: Column name in count_data.obs for stratified
        normalization. Computes separate target sums per group
    :type stratification_column: str, optional
    :param size_factor_cap: Maximum allowed size factor value. Size factors
        exceeding this are clipped to prevent over-normalization
    :type size_factor_cap: numeric, optional
    :param depth_by_sampling: If True, normalize by random sampling with replacement
        to target_sum instead of division. Maintains count distribution
    :type depth_by_sampling: bool, optional
    :param layer: Name of layer to normalize. 'X' for main layer, or any key in
        count_data.layers
    :type layer: str, optional
    :param random_state: Random seed for sampling-based normalization
    :type random_state: int, optional
    :return: Tuple of (normalized AnnData object, gene scale factors or None)
    :rtype: tuple(ad.AnnData, np.ndarray or None)
    """

    # Warn if conflicting parameters are provided
    if (
        (
        target_sum is not None or
        stratification_column is not None or
        size_factor_cap is not None
        )
        and size_factor is not None
    ):
        warnings.warn(
            "target_sum, stratification_column, and size_factor_cap "
            "have no effect when size_factor is passed"
        )

    if (
        (
        size_factor is not None or
        size_factor_cap is not None
        )
        and depth_by_sampling
    ):
        warnings.warn(
            "size_factor and size_factor_cap "
            "have no effect when depth_by_sampling is True"
        )

    # Get reference to the layer data (either X or a specific layer)
    lref = _get_layer(count_data, layer)

    # Compute size factors based on provided parameters
    if subset_genes_for_depth is not None and size_factor is None:
        # Calculate size factors using only a subset of genes (e.g., excluding spike-ins)
        sub_counts, size_factor, target_sum = size_factors(
            _get_layer(count_data[:, subset_genes_for_depth], layer),
            target_sum=target_sum,
            adata=count_data,
            stratification_col=stratification_column,
            size_factor_cap=size_factor_cap
        )
        # Still compute total counts across all genes for reference
        counts = array_sum(lref, 1)
        count_data.obs[f'{layer}_subset_counts'] = sub_counts

    elif size_factor is None:
        # Calculate size factors using all genes
        counts, size_factor, target_sum = size_factors(
            lref,
            target_sum=target_sum,
            adata=count_data,
            stratification_col=stratification_column,
            size_factor_cap=size_factor_cap
        )

    else:
        # Size factors were provided; just compute total counts
        counts = array_sum(lref, 1)

    # Store computed statistics in observation metadata
    count_data.obs[f'{layer}_counts'] = counts
    count_data.obs[f'{layer}_size_factor'] = size_factor

    if target_sum is not None:
        count_data.obs[f'{layer}_target_sum'] = target_sum

    # Apply depth normalization using the appropriate method
    if depth_by_sampling:
        # Normalize by multinomial sampling to target_sum
        _normalize_by_sampling(
            lref,
            target_sum=target_sum,
            random_state=random_state
        )

    elif is_csr(lref) or is_csc(lref):
        # Use optimized sparse normalization for CSR/CSC matrices (in-place)
        from scself.sparse.math import sparse_normalize_total
        sparse_normalize_total(
            lref,
            size_factor=size_factor
        )

    elif layer == 'X':
        # Normalize main layer and reassign to handle potential dtype changes
        count_data.X = _normalize_total(
            lref,
            size_factor=size_factor
        )

    else:
        # Normalize named layer and reassign
        count_data.layers[layer] = _normalize_total(
            lref,
            size_factor=size_factor
        )

    # Apply log transformation if requested
    if log:
        log1p(
            _get_layer(count_data, layer)
        )

    # Apply robust scaling if requested
    if scale:
        lref = _get_layer(count_data, layer)

        scaler = TruncRobustScaler(with_centering=False)

        if scale_factor is None:
            # Fit scaler to compute gene-wise scale factors
            scaler.fit(lref)
            scale_factor = scaler.scale_
        else:
            # Use provided scale factors
            scaler.scale_ = scale_factor

        # Apply scaling using the appropriate method for the data format
        if is_csr(lref):
            # Use optimized sparse column normalization for CSR matrices (in-place)
            from scself.sparse.math import sparse_normalize_columns
            sparse_normalize_columns(
                lref,
                scaler.scale_
            )
        elif layer == 'X':
            # Scale main layer and reassign
            count_data.X = scaler.transform(
                lref
            )
        else:
            # Scale named layer and reassign
            count_data.layers[layer] = scaler.transform(
                lref
            )

        # Store scale factors in variable metadata
        count_data.var[f'{layer}_scale_factor'] = scaler.scale_
    else:
        scale_factor = None

    # Store standardization parameters in unstructured metadata for reproducibility
    count_data.uns['standardization'] = {
        'log': log,
        'scale': scale,
        'target_sum': target_sum,
        'stratification_column': stratification_column,
        'size_factor_cap': size_factor_cap,
        'depth_by_sampling': depth_by_sampling,
        'random_state': random_state
    }

    return count_data, scale_factor


def standardize_data(
    count_data,
    target_sum=None,
    method='log',
    scale_factor=None,
    size_factor=None,
    subset_genes_for_depth=None,
    stratification_column=None,
    size_factor_cap=None,
    depth_by_sampling=False,
    random_state=100,
    layer='X'
):
    """
    Standardize single-cell count data using various normalization methods.

    This is the main entry point for data standardization. It provides a simple
    interface to common normalization workflows. All operations are performed
    in-place, modifying the input AnnData object directly.

    :param count_data: AnnData object containing count data to standardize
    :type count_data: ad.AnnData
    :param target_sum: Target sum for depth normalization. If None, uses the
        median of total counts per cell. Ignored when size_factor is provided.
    :type target_sum: numeric, optional
    :param method: Standardization method to apply. Options are:
        - 'log': Depth normalization + log1p transformation
        - 'scale': Depth normalization + robust scaling
        - 'log_scale': Depth normalization + log1p + robust scaling
        - 'depth': Depth normalization only
        - None: Returns data unmodified
        Defaults to 'log'
    :type method: str or None, optional
    :param scale_factor: Pre-computed gene-wise scale factors for 'scale' methods.
        If None, scale factors are fitted using TruncRobustScaler. Only used when
        method includes scaling.
    :type scale_factor: np.ndarray, optional
    :param size_factor: Pre-computed cell-wise size factors for depth normalization.
        If provided, target_sum and stratification_column have no effect.
    :type size_factor: np.ndarray, optional
    :param subset_genes_for_depth: Boolean mask or indices selecting genes to use
        for computing total counts. Useful for excluding spike-ins or mitochondrial
        genes from normalization factor calculation.
    :type subset_genes_for_depth: array-like, optional
    :param stratification_column: Column name in count_data.obs for stratified
        normalization. Computes separate target sums for each group in this column.
    :type stratification_column: str, optional
    :param size_factor_cap: Maximum allowed size factor value. Size factors above
        this threshold are clipped to prevent over-normalization of low-count cells.
    :type size_factor_cap: numeric, optional
    :param depth_by_sampling: If True, normalize by multinomial sampling to target_sum
        instead of division. Preserves the discrete count distribution.
    :type depth_by_sampling: bool, optional
    :param random_state: Random seed for sampling-based normalization. Only used
        when depth_by_sampling is True.
    :type random_state: int, optional
    :param layer: Name of the layer to standardize. Use 'X' for the main layer,
        or any key in count_data.layers for other layers.
    :type layer: str, optional
    :raises ValueError: If method is not one of the supported options
    :return: Tuple of (standardized AnnData object, gene scale factors or None).
        Scale factors are only returned when method includes scaling.
    :rtype: tuple(ad.AnnData, np.ndarray or None)
    """

    if method == 'log':
        return _normalize(
            count_data,
            target_sum=target_sum,
            log=True,
            size_factor=size_factor,
            subset_genes_for_depth=subset_genes_for_depth,
            layer=layer,
            size_factor_cap=size_factor_cap,
            stratification_column=stratification_column,
            depth_by_sampling=depth_by_sampling,
            random_state=random_state
        )
    elif method == 'scale':
        return _normalize(
            count_data,
            target_sum=target_sum,
            scale=True,
            scale_factor=scale_factor,
            size_factor=size_factor,
            subset_genes_for_depth=subset_genes_for_depth,
            layer=layer,
            size_factor_cap=size_factor_cap,
            stratification_column=stratification_column,
            depth_by_sampling=depth_by_sampling,
            random_state=random_state
        )
    elif method == 'log_scale':
        return _normalize(
            count_data,
            target_sum=target_sum,
            log=True,
            scale=True,
            scale_factor=scale_factor,
            size_factor=size_factor,
            subset_genes_for_depth=subset_genes_for_depth,
            layer=layer,
            size_factor_cap=size_factor_cap,
            stratification_column=stratification_column,
            depth_by_sampling=depth_by_sampling,
            random_state=random_state
        )
    elif method == 'depth':
        return _normalize(
            count_data,
            target_sum=target_sum,
            size_factor=size_factor,
            subset_genes_for_depth=subset_genes_for_depth,
            layer=layer,
            size_factor_cap=size_factor_cap,
            stratification_column=stratification_column,
            depth_by_sampling=depth_by_sampling,
            random_state=random_state
        )
    elif method is None:
        return count_data, None
    else:
        raise ValueError(
            'method must be None, `depth`, `log`, `scale`, or `log_scale`, '
            f'{method} provided'
        )


def _normalize_total(
    data,
    target_sum=None,
    size_factor=None
):
    """
    Normalize data to total counts using size factors.

    Divides each cell (row) by its size factor to normalize for sequencing depth.
    Handles both sparse and dense arrays efficiently.

    :param data: Count matrix to normalize (cells x genes)
    :type data: np.ndarray or scipy.sparse matrix
    :param target_sum: Target sum for normalization. Only used if size_factor is None.
    :type target_sum: numeric, optional
    :param size_factor: Pre-computed size factors for each cell. If None, computed
        from data using target_sum.
    :type size_factor: np.ndarray, optional
    :return: Normalized data matrix
    :rtype: Same type as input (np.ndarray or scipy.sparse matrix)
    """

    if size_factor is None:
        _, size_factor, _ = _size_factors_all(data, target_sum=target_sum)

    if sps.issparse(data):
        # For sparse matrices, use matrix multiplication to normalize
        # Can't really
        return sps.diags_array(
            1/size_factor,
            format='csr'
        ) @ data
    else:
        # For dense arrays, convert to float and divide in-place
        cast_to_float_inplace(data)
        return np.divide(data, size_factor[:, None], out=data)


def _normalize_by_sampling(
    data,
    target_sum,
    random_state=100
):
    """
    Normalize count data by multinomial sampling.

    Instead of dividing by size factors, this function normalizes by randomly
    sampling counts to a target depth. This preserves the discrete count
    distribution and can be more appropriate for certain downstream analyses.
    Operations are performed in-place.

    :param data: Count matrix to normalize (cells x genes). Must be CSR sparse
        or dense array.
    :type data: np.ndarray or scipy.sparse.csr_matrix
    :param target_sum: Target total counts per cell. Can be a scalar (same for
        all cells) or array-like (different per cell).
    :type target_sum: numeric or array-like
    :param random_state: Random seed for reproducibility
    :type random_state: int, optional
    :raises RuntimeError: If data is sparse but not in CSR format
    """

    rng = np.random.default_rng(random_state)

    # Convert scalar target_sum to array
    if not isinstance(target_sum, (np.ndarray, list, tuple)):
        target_sum = np.full(data.shape[0], target_sum, dtype=int)

    if is_csr(data):
        # Process each cell (row) in the CSR matrix
        for row in range(data.shape[0]):
            _ind = data.indices[data.indptr[row]:data.indptr[row + 1]]
            _data = data.data[data.indptr[row]:data.indptr[row + 1]]
            _p = _data / _data.sum()

            # Sample with replacement according to the count distribution
            _data[:] = np.bincount(
                rng.choice(
                    np.arange(_ind.shape[0]),
                    size=target_sum[row],
                    replace=True,
                    p=_p
                ),
                minlength=_ind.shape[0]
            )

        # Remove any zeros created by sampling and convert to float
        data.eliminate_zeros()
        cast_to_float_inplace(data.data)

    elif sps.issparse(data):
        raise RuntimeError("For sampling in place, data must be CSR or dense")
    else:
        # Process each cell (row) in the dense array
        for row in range(data.shape[0]):
            _p = data[row] / data[row].sum()

            # Sample with replacement according to the count distribution
            data[row, :] = np.bincount(
                rng.choice(
                    np.arange(data.shape[1]),
                    size=target_sum[row],
                    replace=True,
                    p=_p
                ),
                minlength=data.shape[1]
            )

        cast_to_float_inplace(data)


def size_factors(
    data,
    target_sum,
    adata=None,
    stratification_col=None,
    size_factor_cap=None
):
    """
    Compute size factors for depth normalization.

    This function computes cell-wise size factors for normalizing sequencing depth.
    Supports both uniform normalization (all cells to same target) and stratified
    normalization (different targets per group).

    :param data: Count matrix (cells x genes)
    :type data: np.ndarray or scipy.sparse matrix
    :param target_sum: Target sum for normalization. If None, uses median of total
        counts. For stratified normalization, must be dict or pd.Series mapping
        categories to target depths.
    :type target_sum: numeric, dict, or pd.Series
    :param adata: AnnData object containing the data. Required for stratified
        normalization to access observation metadata.
    :type adata: ad.AnnData, optional
    :param stratification_col: Column name in adata.obs for grouping cells. If
        provided, computes separate size factors for each group.
    :type stratification_col: str, optional
    :param size_factor_cap: Minimum size factor value (applied as lower bound after
        clipping). Size factors below this are set to this value.
    :type size_factor_cap: numeric, optional
    :raises ValueError: If stratification_col is provided without adata, or if
        target_sum is not dict/Series when using stratification
    :return: Tuple of (total counts array, size factors array, target sum array/value)
    :rtype: tuple(np.ndarray, np.ndarray, numeric or np.ndarray)
    """

    if stratification_col is None:
        # Compute uniform size factors for all cells
        sf = _size_factors_all(
            data,
            target_sum
        )

    elif adata is not None and stratification_col is not None:
        # Validate target_sum format for stratified normalization
        if (
            target_sum is not None and
            not isinstance(target_sum, (dict, pd.Series))
        ):
            raise ValueError(
                "target_sum must be a dict or pd.Series "
                "keying categories to the target depth "
                "if stratification_col is not None"
            )

        # Compute stratified size factors
        sf =  _size_factors_stratified(
            data,
            adata,
            stratification_col,
            target_sum=target_sum
        )

    else:
        raise ValueError("provide both adata and stratification_col")

    # Apply lower bound to size factors if specified
    if size_factor_cap is not None:
        np.clip(sf[1], size_factor_cap, None, out=sf[1])

    return sf


def _size_factors_all(
    data,
    target_sum=None,
):
    """
    Compute uniform size factors for all cells.

    Calculates size factors as the ratio of each cell's total counts to the target sum.
    This is the standard approach for depth normalization in single-cell analysis.

    :param data: Count matrix (cells x genes)
    :type data: np.ndarray or scipy.sparse matrix
    :param target_sum: Target sum for normalization. If None, uses median of
        total counts across all cells.
    :type target_sum: numeric, optional
    :return: Tuple of (total counts per cell, size factors per cell, target sum used)
    :rtype: tuple(np.ndarray, np.ndarray, numeric)
    """
    counts = array_sum(data, 1)

    if target_sum is None:
        target_sum = np.median(counts)

    # Size factor = observed_counts / target_counts
    size_factor = counts / target_sum
    # Prevent division by zero for empty cells
    size_factor[counts == 0] = 1.

    return counts, size_factor, target_sum


def _size_factors_stratified(
    data,
    adata,
    stratification_col,
    target_sum=None
):
    """
    Compute stratified size factors with group-specific targets.

    Calculates size factors separately for each group defined by a categorical
    variable. This is useful when different cell types or experimental conditions
    have systematically different sequencing depths.

    :param data: Count matrix (cells x genes)
    :type data: np.ndarray or scipy.sparse matrix
    :param adata: AnnData object containing observation metadata
    :type adata: ad.AnnData
    :param stratification_col: Column name in adata.obs defining cell groups
    :type stratification_col: str
    :param target_sum: Group-specific target sums. If None, uses median counts
        within each group. If provided, must be dict or Series mapping group
        labels to target sums.
    :type target_sum: dict, pd.Series, or None, optional
    :return: Tuple of (total counts per cell, size factors per cell,
        target sums per cell as array)
    :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
    """

    counts = array_sum(data, 1)

    # Create DataFrame with stratification column and counts
    size_factor = adata.obs[[stratification_col]].copy()
    size_factor['counts'] = counts

    if target_sum is None:
        # Compute median counts for each group
        target_sum = size_factor.groupby(
            stratification_col,
            observed=True
        ).agg('median')

        try:
            target_sum = target_sum.to_frame()
        except AttributeError:
            pass

        target_sum = target_sum.rename(
            {'counts': 'medians'},
            axis=1
        )

    else:
        # Use provided group-specific target sums
        target_sum = pd.Series(target_sum).rename('medians')

    # Join target sums to each cell based on its group membership
    size_factor = size_factor.join(target_sum, on=stratification_col)
    target_sum = size_factor['medians'].values.astype(int)

    # Calculate size factors: observed_counts / group_target
    size_factor = size_factor['counts'] / size_factor['medians']
    # Prevent division by zero for empty cells
    size_factor[size_factor == 0] = 1.0

    return counts, size_factor.values, target_sum


def log1p(data):
    """
    Apply log(1 + x) transformation in-place.

    Computes the natural logarithm of one plus each element. This transformation
    is commonly used in single-cell analysis to stabilize variance and make the
    data more normally distributed.

    :param data: Data to transform. Can be dense or sparse matrix. Modified in-place.
    :type data: np.ndarray or scipy.sparse matrix
    :return: Reference to the transformed data
    :rtype: np.ndarray (data array for sparse, or input array for dense)
    """

    if sps.issparse(data):
        # For sparse matrices, transform only the data values
        data = data.data

    # Apply log1p transformation in-place
    np.log1p(data, out=data)

    return data


def _get_layer(adata, layer):
    """
    Get a reference to a specific data layer in an AnnData object.

    :param adata: AnnData object
    :type adata: ad.AnnData
    :param layer: Layer name. Use 'X' for main layer, or any key from adata.layers.
    :type layer: str
    :return: Reference to the requested layer
    :rtype: np.ndarray or scipy.sparse matrix
    """
    if layer == 'X':
        return adata.X
    else:
        return adata.layers[layer]
