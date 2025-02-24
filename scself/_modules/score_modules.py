import numpy as np

from scself.scaling import TruncMinMaxScaler
from scself.utils.correlation import corrcoef

def score_all_modules(
    adata,
    modules=None,
    module_column='gene_module',
    output_key_suffix='_score',
    obs_mask=None,
    layer='X',
    scaler=TruncMinMaxScaler(),
    fit_scaler=True,
    clipping=None,
    **kwargs
):
    """
    Score all modules in an AnnData object by calculating module scores
    for each set of genes.

    :param adata: AnnData object containing gene expression data
    :type adata: ad.AnnData
    :param modules: List of module names to score. If None,
        uses all unique values in module_column
    :type modules: list, optional
    :param module_column: Column in adata.var containing module assignments
    :type module_column: str, optional
    :param output_key_suffix: Suffix for output keys
        in adata.obsm and adata.uns
    :type output_key_suffix: str, optional
    :param obs_mask: Boolean mask or slice for observations to score
    :type obs_mask: np.ndarray or slice, optional
    :param layer: Layer in adata to use for scoring
    :type layer: str, optional
    :param scaler: Scaler object to use for scaling gene expression.
        Must have fit() and transform() methods
    :type scaler: object, optional
    :param fit_scaler: Whether to fit the scaler on the data. If False,
        uses pre-fit scaler
    :type fit_scaler: bool, optional
    :param clipping: Tuple of (min, max) values to clip scores to
    :type clipping: tuple, optional
    :param kwargs: Additional keyword arguments passed to module_score()

    :return: Original AnnData with added fields:
        - adata.obsm[]: Module scores matrix (n_cells x n_modules)
        - adata.uns[]: List of module names
        - adata.uns[]: Correlation matrix between module scores
        - adata.obs[]: Index of highest scoring module per cell
    :rtype: ad.AnnData
    """

    
    _outkey = f'{module_column}{output_key_suffix}'
    
    if modules is None:
        modules = [
            x
            for x in adata.var[module_column].unique()
            if x not in [-1, '-1']
        ]
    
    if obs_mask is None:
        obs_mask = slice(None)
    
    adata.obsm[_outkey] = np.full(
        (adata.shape[0], len(modules)),
        np.nan
    )

    for i, cat in enumerate(modules):
        adata.obsm[_outkey][obs_mask, i] = module_score(
            adata,
            adata.var_names[adata.var[module_column] == cat],
            obs_mask=obs_mask,
            layer=layer,
            scaler=scaler,
            fit_scaler=fit_scaler,
            clipping=clipping,
            **kwargs
        )
    
    adata.uns[_outkey] = modules
    adata.uns[_outkey + '_corrcoef'] = corrcoef(adata.obsm[_outkey][obs_mask, :])
    adata.obs[_outkey + "_max"] = np.argmax(adata.obsm[_outkey], axis=1)

    return adata

def score_all_submodules(
    adata,
    modules=None,
    submodules=None,
    module_column='gene_module',
    submodule_column='gene_submodule',
    output_key_suffix='_score',
    obs_mask=None,
    layer='X',
    scaler=TruncMinMaxScaler(),
    fit_scaler=True,
    clipping=None,
    **kwargs
):
    """
    Score all modules in an AnnData object by calculating module scores
    for each set of genes.

    :param adata: AnnData object containing gene expression data
    :type adata: ad.AnnData
    :param modules: List of module names to score. If None,
        uses all unique values in module_column
    :type modules: list, optional
    :param module_column: Column in adata.var containing module assignments
    :type module_column: str, optional
    :param output_key_suffix: Suffix for output keys
        in adata.obsm and adata.uns
    :type output_key_suffix: str, optional
    :param obs_mask: Boolean mask or slice for observations to score
    :type obs_mask: np.ndarray or slice, optional
    :param layer: Layer in adata to use for scoring
    :type layer: str, optional
    :param scaler: Scaler object to use for scaling gene expression.
        Must have fit() and transform() methods
    :type scaler: object, optional
    :param fit_scaler: Whether to fit the scaler on the data. If False,
        uses pre-fit scaler
    :type fit_scaler: bool, optional
    :param clipping: Tuple of (min, max) values to clip scores to
    :type clipping: tuple, optional
    :param kwargs: Additional keyword arguments passed to module_score()

    :return: Original AnnData with added fields:
        - adata.obsm[]: Module scores matrix (n_cells x n_modules)
        - adata.uns[]: List of module names
        - adata.uns[]: Correlation matrix between module scores
        - adata.obs[]: Index of highest scoring module per cell
    :rtype: ad.AnnData
    """

    
    _outkey = f'{submodule_column}{output_key_suffix}'
    
    # Get common modules
    if modules is None:
        modules = [
            x
            for x in adata.var[module_column].unique()
            if x not in [-1, '-1']
        ]

    if submodules is None:
        if submodules is None:
            submodules = np.array(
                [
                    x for x in adata.var[
                        [module_column, submodule_column]
                    ].value_counts().sort_index().index.values
                    if (x[0] not in [-1, '-1']) and (x[0] in adata.uns[module_column + "_score"])
                ],
                dtype=[('module', int), ('submodule', int)]
            )

    
    if obs_mask is None:
        obs_mask = slice(None)
    
    adata.obsm[_outkey] = np.full(
        (adata.shape[0], len(submodules)),
        np.nan
    )

    for i, (cat, subcat) in enumerate(submodules):
        _idx = adata.var[module_column] == cat
        _idx &= adata.var[submodule_column] == subcat

        adata.obsm[_outkey][obs_mask, i] = module_score(
            adata,
            adata.var_names[_idx],
            obs_mask=obs_mask,
            layer=layer,
            scaler=scaler,
            fit_scaler=fit_scaler,
            clipping=clipping,
            **kwargs
        )
    
    adata.uns[_outkey] = submodules
    adata.uns[_outkey + '_corrcoef'] = corrcoef(adata.obsm[_outkey][obs_mask, :])
    adata.obs[_outkey + "_max"] = np.argmax(adata.obsm[_outkey], axis=1)

    return adata

def module_score(
    adata,
    genes,
    layer='X',
    scaler=TruncMinMaxScaler(),
    fit_scaler=True,
    clipping=None,
    obs_mask=None,
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

    if obs_mask is None:
        obs_mask = slice(None)

    if len(genes) == 0:
        return np.full((dref.shape[0],), np.nan)[obs_mask]

    _data = dref[:, adata.var_names.isin(genes)][obs_mask, :]

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
