import sys

from scipy.cluster.hierarchy import (
    dendrogram,
    linkage
)
from scipy.spatial.distance import pdist


def hclust(data, metric='euclidean', method='ward'):

    if data.shape > 10000:
        sys.setrecursionlimit(10000)

    return dendrogram(
        linkage(
            pdist(data, metric=metric),
            method=method
        ),
        no_plot=True
    )["leaves"]
