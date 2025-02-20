# scself

[![PyPI version](https://badge.fury.io/py/scself.svg)](https://badge.fury.io/py/scself)

Self Supervised Tools for Single Cell Data

Molecular Cross-Validation for PCs [arXiv manuscript](https://www.biorxiv.org/content/10.1101/786269v1)

```
mcv(
    count_data,
    n=1,
    n_pcs=100,
    random_seed=800,
    p=0.5,
    metric='mse',
    standardization_method='log',
    metric_kwargs={},
    silent=False,
    verbose=None,
    zero_center=False
)
```

Noise2Self for kNN selection [arXiv manuscript](https://arxiv.org/abs/1901.11365)

```
noise2self(
    count_data,
    neighbors=None,
    npcs=None,
    metric='euclidean',
    loss='mse',
    loss_kwargs={},
    return_errors=False,
    connectivity=False,
    standardization_method='log',
    pc_data=None,
    chunk_size=10000,
    verbose=None
)
```

Implemented as in [DEWÄKSS](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008569)

Feature module and submodule determination using pearson correlation distance, kNN embedding, and leiden clustering
```
get_correlation_modules(
    adata,
    layer='X',
    n_neighbors=10,
    leiden_kwargs={},
    output_key='gene_module'
)

get_correlation_submodules(
    adata,
    layer='X',
    n_neighbors=10,
    leiden_kwargs={},
    input_key='gene_module',
    output_key='gene_submodule'
)
```