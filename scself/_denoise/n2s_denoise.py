import numpy as np
import scipy.sparse as sps

from ..utils import dot
from .._noise2self.common import row_normalize
from .._noise2self.graph import combine_row_stochastic_graphs


def denoise_data(
    data,
    graphs,
    connectivity=False,
    zero_threshold=None,
    chunk_size=None
):

    data, graphs, n_modes, n_obs = _check_inputs(data, graphs)

    # Not chunked
    if chunk_size is None:
        _graph = _combine_graphs(graphs, connectivity=connectivity)
        _denoised = [
            _denoise_chunk(
                d,
                _graph,
                zero_threshold=zero_threshold
            )
            for d in data
        ]

    # Chunked writing into preallocated dense arrays
    elif zero_threshold is None:
        _denoised = [
            np.zeros(d.shape, dtype=d.dtype)
            for d in data
        ]

        for start, end in _chunk_gen(chunk_size, n_obs):
            _graph_chunk = _combine_graphs(
                [g[start:end] for g in graphs],
                connectivity=connectivity
            )

            for i in range(n_modes):
                _denoise_chunk(
                    data[i],
                    _graph_chunk,
                    out=_denoised[i][start:end]
                )

    # Chunked with separate arrays for each chunk
    # stacked at the end
    else:
        _denoised = [[] for i in range(n_modes)]

        for start, end in _chunk_gen(chunk_size, n_obs):
            _graph_chunk = _combine_graphs(
                [g[start:end] for g in graphs],
                connectivity=connectivity
            )

            for i in range(n_modes):
                _denoised[i].append(
                    _denoise_chunk(
                        data[i],
                        _graph_chunk,
                        zero_threshold=zero_threshold
                    )
                )

        _denoised = [
            sps.vstack(d)
            if sps.issparse(d[0])
            else np.vstack(d)
            for d in _denoised
        ]

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
        return combine_row_stochastic_graphs(
            [
                row_normalize(g, connectivity=connectivity)
                for g in graph_chunks
            ]
        )

    else:
        return row_normalize(graph_chunks, connectivity=connectivity)


def _denoise_chunk(
    x,
    graph,
    zero_threshold=None,
    out=None
):

    out = dot(
        graph,
        x,
        out=out,
        dense=zero_threshold is None
    )

    if zero_threshold is not None:

        if sps.issparse(out):
            out.data[out.data < zero_threshold] = 0
        else:
            out[out < zero_threshold] = 0

        try:
            out.eliminate_zeros()
        except AttributeError:
            pass

    return out
