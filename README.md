# scself

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
def noise2self(
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