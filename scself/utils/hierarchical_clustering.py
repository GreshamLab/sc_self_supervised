import sys

from scipy.cluster.hierarchy import (
    dendrogram,
    linkage,
    fcluster
)
from scipy.spatial.distance import pdist


def hclust(data, metric='euclidean', method='ward', return_fcluster=False, **kwargs):

    # Increase recursion limit so dendrogram doesn't throw a fit
    # if needed
    if (data.shape[0] > 10000) and (sys.getrecursionlimit() < 10000):
        sys.setrecursionlimit(10000)

    # Turn pandas into numpy
    try:
        data = data.values
    except AttributeError:
        pass

    # Reshape a 1d vector into a column vector for pdist
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    # Get linkage matrix
    _links = linkage(
        pdist(data, metric=metric),
        method=method
    )

    # Use dendrogram for ordering
    _order = dendrogram(
        _links,
        no_plot=True
    )["leaves"]

    if return_fcluster:
        return _order, fcluster(
            _links,
            **kwargs
        )
    else:
        return _order
