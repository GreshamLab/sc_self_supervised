import numpy as np
import scipy.sparse as sps

from ..utils import dot, shrink_array_to_zero_inplace
from .._noise2self.common import row_normalize
from .._noise2self.graph import combine_row_stochastic_graphs


def denoise_data(
    data,
    graphs,
    connectivity=False,
    zero_threshold=None,
    chunk_size=None,
    dense=None
):
    """
    Denoise data using a kNN graph.

    Each argument can also be provided as a list where each
    value represents a different data modality.

    Returns a denoised array or a list of denoised arrays
    if multiple data modalities is provided.

    :param data: Data to be denoised
    :type data: list, np.ndarray, sp.spmatrix, sp.sparse_array
    :param graphs: kNN graph to use for denoising
    :type graphs: list, np.ndarray, sp.spmatrix, sp.sparse_array
    :param connectivity: Use kNN graph as a connectivity graph,
        instead of a distance graph. Defaults to False
    :type connectivity: bool, optional
    :param zero_threshold: Shrink values smaller in absolute value
        down to zero, defaults to None
    :type zero_threshold: list, float, optional
    :param chunk_size: Size of chunks (# observations),
        can reduce memory usage. Defaults to None
    :type chunk_size: list, None, int, optional
    :param dense: Return a dense result instead of sparse,
        None returns sparse if input data is sparse and dense
        if input data is dense. Defaults to None
    :type dense: list, bool, optional
    :return: Returns denoised data as sparse or dense.
        Will return a list if multiple data modes are provided in
        the data argument
    :rtype: list, np.ndarray, sp.spmatrix
    """

    data, graphs, n_modes, n_obs = _check_inputs(data, graphs)

    if dense is None:
        dense = [not sps.issparse(d) for d in data]
    elif not isinstance(dense, (tuple, list)):
        dense = [dense] * n_modes

    if zero_threshold is None:
        zero_threshold = [None] * n_modes
    elif not isinstance(zero_threshold, (tuple, list)):
        zero_threshold = [zero_threshold] * n_modes

    # Not chunked
    if chunk_size is None or (chunk_size >= n_obs):
        _graph = _combine_graphs(graphs, connectivity=connectivity)
        _denoised = [
            _denoise_chunk(
                d,
                _graph,
                zero_threshold=zero_threshold[i],
                dense=dense[i]
            )
            for i, d in enumerate(data)
        ]

    # Preallocate dense arrays to write into chunk-wise
    # and keep a list of sparse chunks to be vstacked
    else:

        # Preallocate
        _denoised = [
            np.zeros(d.shape, dtype=d.dtype)
            if dense[i]
            else []
            for i, d in enumerate(data)
        ]

        for start, end in _chunk_gen(chunk_size, n_obs):

            # Process chunk combined graph
            _graph_chunk = _combine_graphs(
                [g[start:end] for g in graphs],
                connectivity=connectivity
            )

            # Process chunk for each modality
            for i in range(n_modes):
                _dref = _denoise_chunk(
                    data[i],
                    _graph_chunk,
                    out=_denoised[i][start:end] if dense[i] else None,
                    zero_threshold=zero_threshold[i]
                )

                if not dense[i]:
                    _denoised[i].append(_dref)

        # Stack sparse arrays if needed
        for i in range(n_modes):
            if not dense[i]:
                _denoised[i] = sps.vstack(_denoised[i])

    if len(_denoised) == 1:
        return _denoised[0]
    else:
        return _denoised


def _chunk_gen(chunk_size, n_obs):

    _n_chunks = int(n_obs / chunk_size) + 1

    for i in range(_n_chunks):
        _start = i * chunk_size
        _end = min(n_obs, (i + 1) * chunk_size)

        if _end <= _start:
            return

        yield _start, _end


def _check_inputs(data, graphs):

    if isinstance(data, (tuple, list)):
        _n_modes = len(data)
        _n_obs = data[0].shape[0]

        for d in data:
            if d.shape[0] != _n_obs:
                raise ValueError(
                    "Data objects have different numbers of observations: "
                    f"{[d.shape[0] for d in data]}"
                )
    else:
        _n_modes = 1
        _n_obs = data.shape[0]
        data = [data]

    if isinstance(graphs, (tuple, list)):
        for g in graphs:
            if g.shape[0] != _n_obs:
                raise ValueError(
                    "Data objects have different numbers of observations: "
                    f"{[g.shape[0] for g in graphs]}"
                )

    elif (
        (graphs.shape[0] != _n_obs) or
        (graphs.shape[1] != _n_obs)
    ):
        raise ValueError(
            "Data objects and graphs are not compatible shapes: "
            f"{graphs.shape}"
        )

    else:
        graphs = [graphs]

    return data, graphs, _n_modes, _n_obs


def _combine_graphs(
    graph_chunks,
    connectivity=False
):

    if isinstance(graph_chunks, (tuple, list)):

        if not isinstance(connectivity, (tuple, list)):
            connectivity = [connectivity] * len(graph_chunks)

        return combine_row_stochastic_graphs(
            [
                row_normalize(g, connectivity=connectivity[i])
                for i, g in enumerate(graph_chunks)
            ]
        )

    else:
        return row_normalize(graph_chunks, connectivity=connectivity)


def _denoise_chunk(
    x,
    graph,
    zero_threshold=None,
    out=None,
    dense=False
):

    out = dot(
        graph,
        x,
        out=out,
        dense=dense
    )

    if zero_threshold is not None:

        if sps.issparse(out):
            shrink_array_to_zero_inplace(
                out.data,
                zero_threshold
            )

        else:
            shrink_array_to_zero_inplace(
                out,
                zero_threshold
            )

        try:
            out.eliminate_zeros()
        except AttributeError:
            pass

    return out
