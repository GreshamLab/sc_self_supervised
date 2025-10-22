"""
Optimized mathematical operations for sparse matrices.

This module provides high-performance sparse matrix operations using Numba JIT
compilation. All operations are designed for CSR and CSC sparse matrix formats
and optimize memory usage by working in-place where possible.
"""

import numpy as np
import scipy.sparse as sps
import numba

from scself.utils import cast_to_float_inplace


def is_csr(x):
    """
    Check if a matrix is in CSR (Compressed Sparse Row) format.

    Works with both legacy scipy.sparse.csr_matrix and newer scipy.sparse.csr_array.

    :param x: Matrix to check
    :type x: any
    :return: True if x is a CSR matrix or array
    :rtype: bool
    """
    return sps.isspmatrix_csr(x) or isinstance(x, sps.csr_array)


def is_csc(x):
    """
    Check if a matrix is in CSC (Compressed Sparse Column) format.

    Works with both legacy scipy.sparse.csc_matrix and newer scipy.sparse.csc_array.

    :param x: Matrix to check
    :type x: any
    :return: True if x is a CSC matrix or array
    :rtype: bool
    """
    return sps.isspmatrix_csc(x) or isinstance(x, sps.csc_array)


def mcv_mean_error_sparse(
    x,
    pc,
    rotation,
    axis=1,
    squared=False,
    **metric_kwargs
):
    """
    Calculate mean reconstruction error between sparse data and PCA projection.

    This function computes the error between a sparse matrix and its reconstruction
    from principal components. It processes data row-wise to minimize memory usage,
    which is crucial for large sparse matrices. The error can be computed as either
    mean absolute error (MAE) or mean squared error (MSE).

    :param x: Sparse data matrix in CSR format
    :type x: sp.sparse.spmatrix, sp.sparse.sparray
    :param pc: Principal component scores for each observation (cells x components)
    :type pc: np.ndarray
    :param rotation: PCA rotation matrix mapping PCs back to features (components x features)
    :type rotation: np.ndarray
    :param axis: Aggregation axis for error calculation:
        - 1: Row-wise error (one value per observation)
        - 0: Column-wise error (one value per feature)
        - None: Overall mean error (single scalar)
        Defaults to 1.
    :type axis: int or None, optional
    :param squared: If True, calculate mean squared error (MSE).
        If False, calculate mean absolute error (MAE). Defaults to False.
    :type squared: bool, optional
    :param metric_kwargs: Additional keyword arguments (currently unused, for compatibility)
    :raises ValueError: If axis is not 0, 1, or None
    :return: Mean error over specified axis (array) or overall mean (scalar)
    :rtype: np.ndarray or float
    """

    # Select appropriate computation function based on aggregation axis
    if axis == 1:
        func = _mean_error_rowwise
    elif axis == 0:
        func = _mean_error_columnwise
    elif axis is None:
        func = _mean_error_rowwise
    else:
        raise ValueError

    # Compute reconstruction error using JIT-compiled function
    # Ensure arrays are contiguous for optimal Numba performance
    y = func(
        x.data,
        x.indices,
        x.indptr,
        np.ascontiguousarray(pc),
        np.ascontiguousarray(rotation, dtype=pc.dtype),
        x.shape[1],
        squared
    )

    # If flattened output requested, compute overall mean
    if axis is None:
        y = y.mean()

    return y


def sparse_sum(sparse_array, axis=None, squared=False):
    """
    Compute sum of sparse array elements along specified axis.

    This function provides optimized summation for CSR and CSC sparse matrices
    using Numba-compiled functions. It's more efficient than scipy's built-in
    sum for large matrices.

    :param sparse_array: Sparse matrix to sum
    :type sparse_array: scipy.sparse matrix (CSR or CSC)
    :param axis: Axis along which to sum:
        - None: Sum all elements (returns scalar)
        - 0: Sum along rows (returns array of column sums)
        - 1: Sum along columns (returns array of row sums)
        Defaults to None.
    :type axis: int or None, optional
    :param squared: If True, sum squared values. Currently unused but kept
        for API compatibility. Defaults to False.
    :type squared: bool, optional
    :raises ValueError: If input is not a sparse array or format is not CSR/CSC
    :return: Sum of elements (scalar if axis=None, array otherwise)
    :rtype: float or np.ndarray
    """

    if not sps.issparse(sparse_array):
        raise ValueError("sparse_sum requires a sparse array")

    if axis is None and squared:
        # Get sum of squares for a 1d array with np.inner
        return np.inner(sparse_array.data, sparse_array.data)
    elif axis is None:
        # Sum all non-zero elements (most efficient for sparse)
        return np.sum(sparse_array.data)

    elif axis == 0:
        # Sum along rows (column sums)
        if is_csr(sparse_array):
            # For CSR: aggregate by column indices
            return _sum_on_indices(
                sparse_array.data,
                sparse_array.indices,
                sparse_array.shape[1],
                squared=squared
            )
        elif is_csc(sparse_array):
            # For CSC: aggregate by pointer ranges (one per column)
            return _sum_on_indptr(
                sparse_array.data,
                sparse_array.indptr,
                squared=squared
            )
        else:
            raise ValueError

    elif axis == 1:
        # Sum along columns (row sums)
        if is_csr(sparse_array):
            # For CSR: aggregate by pointer ranges (one per row)
            return _sum_on_indptr(
                sparse_array.data,
                sparse_array.indptr,
                squared=squared
            )
        elif is_csc(sparse_array):
            # For CSC: aggregate by row indices
            return _sum_on_indices(
                sparse_array.data,
                sparse_array.indices,
                sparse_array.shape[0],
                squared=squared
            )
        else:
            raise ValueError


def sparse_normalize_columns(sparse_array, column_norm_vec):
    """
    Normalize columns of a CSR sparse matrix by dividing by column-specific factors.

    This function divides each column by its corresponding normalization factor
    in-place. Commonly used for gene-wise scaling in single-cell data. The operation
    converts integer data to float if necessary.

    :param sparse_array: Sparse matrix to normalize in CSR format
    :type sparse_array: scipy.sparse.csr_matrix or scipy.sparse.csr_array
    :param column_norm_vec: Normalization factor for each column. Must have
        length equal to number of columns.
    :type column_norm_vec: np.ndarray
    :raises ValueError: If sparse_array is not in CSR format
    """

    if not is_csr(sparse_array):
        raise ValueError("sparse_sum requires a sparse csr_array")

    # Convert integer data to float for normalization
    if sparse_array.data.dtype == np.int32:
        dtype = np.float32
    elif sparse_array.data.dtype == np.int64:
        dtype = np.float64
    else:
        dtype = None

    if dtype is not None:
        # Use view to convert in-place without copying
        float_view = sparse_array.data.view(dtype)
        float_view[:] = sparse_array.data
        sparse_array.data = float_view

    # Perform column-wise division using Numba-compiled function
    _csr_column_divide(
        sparse_array.data,
        sparse_array.indices,
        column_norm_vec
    )


def sparse_normalize_total(
    sparse_array,
    target_sum=10_000,
    size_factor=None
):
    """
    Normalize rows (cells) of sparse matrix to target total counts.

    This function performs depth normalization on sparse count matrices, dividing
    each row by its size factor to normalize for sequencing depth. Operates in-place
    for memory efficiency. Commonly used for normalizing single-cell RNA-seq data.

    :param sparse_array: Sparse count matrix to normalize (cells x genes)
    :type sparse_array: scipy.sparse.csr_matrix, scipy.sparse.csr_array,
        scipy.sparse.csc_matrix, or scipy.sparse.csc_array
    :param target_sum: Target total counts per row. If None, uses median of
        row counts. Only used if size_factor is not provided. Defaults to 10,000.
    :type target_sum: numeric or None, optional
    :param size_factor: Pre-computed size factors for each row. If None, computed
        as row_counts / target_sum. Defaults to None.
    :type size_factor: np.ndarray or None, optional
    :raises ValueError: If sparse_array is not in CSR or CSC format
    """

    if not (
        is_csr(sparse_array) or
        is_csc(sparse_array)
    ):
        raise ValueError(
            "sparse_normalize_total requires a sparse "
            "csr_array or csc_array"
        )

    # Convert to float for normalization
    cast_to_float_inplace(sparse_array.data)

    # Compute size factors if not provided
    if size_factor is None:
        n_counts = sparse_sum(sparse_array, axis=1)

        if target_sum is None:
            target_sum = np.median(n_counts)

        # Size factor = observed_counts / target_counts
        size_factor = n_counts / target_sum
        # Prevent division by zero for empty rows
        size_factor[n_counts == 0] = 1.

    # Divide rows by size factors using format-appropriate method
    if is_csr(sparse_array):
        # CSR: rows are contiguous, use row division
        _csr_row_divide(
            sparse_array.data,
            sparse_array.indptr,
            size_factor
        )
    elif is_csc(sparse_array):
        # CSC: columns are contiguous, use column-style indexing for rows
        _csr_column_divide(
            sparse_array.data,
            sparse_array.indices,
            size_factor
        )
    else:
        raise ValueError

def sparse_csr_extract_columns(
    sparse_array,
    fake_csc_matrix
):
    """
    Extract and reorganize CSR matrix data into column-major order.

    This function rearranges the data array of a CSR sparse matrix to be ordered
    by columns instead of rows, essentially converting to CSC ordering. Useful for
    column-wise operations on CSR matrices without full format conversion.

    :param sparse_array: CSR sparse matrix to extract columns from
    :type sparse_array: scipy.sparse.csr_matrix or scipy.sparse.csr_array
    :param fake_csc_matrix: If True, return a CSC matrix with reorganized data.
        If False, return raw data and index pointer arrays.
    :type fake_csc_matrix: bool
    :return: Either a CSC matrix (if fake_csc_matrix=True) or tuple of
        (reorganized data array, column index pointer array)
    :rtype: scipy.sparse.csc_matrix or tuple(np.ndarray, np.ndarray)
    """

    # Compute CSC-style column pointers from CSR indices
    col_indptr = _csr_to_csc_indptr(
        sparse_array.indices,
        sparse_array.shape[1]
    )

    # Reorder data array to be column-major
    new_data = _csr_extract_columns(
        sparse_array.data,
        sparse_array.indices,
        col_indptr
    )

    if fake_csc_matrix:
        # Create a CSC matrix shell with the reorganized data
        arr = sps.csc_matrix(
            sparse_array.shape,
            dtype=sparse_array.dtype
        )

        arr.data = new_data
        # Placeholder indices (actual row indices not computed)
        arr.indices = np.zeros((1,), dtype=col_indptr.dtype)
        arr.indptr = col_indptr

        return arr

    else:
        # Return raw arrays for manual processing
        return new_data, col_indptr


@numba.njit(parallel=False)
def _mean_error_rowwise(
    a_data,
    a_indices,
    a_indptr,
    b_pcs,
    b_rotation,
    n_cols,
    squared
):
    """
    Compute row-wise mean reconstruction error (JIT-compiled).

    For each row, reconstructs the data from PCA components and computes the
    mean error (MAE or MSE) between original sparse data and reconstruction.
    Only non-zero elements in the sparse array are subtracted from reconstruction.

    :param a_data: Non-zero values from sparse CSR matrix
    :param a_indices: Column indices for non-zero values
    :param a_indptr: Index pointers for row boundaries
    :param b_pcs: Principal component scores (n_observations x n_components)
    :param b_rotation: PCA rotation matrix (n_components x n_features)
    :param n_cols: Number of columns in original matrix
    :param squared: If True, compute MSE; if False, compute MAE
    :return: Array of mean errors, one per row
    """

    n_row = b_pcs.shape[0]

    output = np.zeros(n_row, dtype=float)

    # Process each row independently
    for i in numba.prange(n_row):

        # Get indices of non-zero elements in this row
        _idx_a = a_indices[a_indptr[i]:a_indptr[i + 1]]
        _nnz_a = _idx_a.shape[0]

        # Reconstruct row from PCA: PC_scores @ rotation
        row = b_pcs[i, :] @ b_rotation

        # Subtract actual sparse values from reconstruction
        if _nnz_a == 0:
            pass  # No non-zero elements, error is just the reconstruction
        else:
            row[_idx_a] -= a_data[a_indptr[i]:a_indptr[i + 1]]

        # Compute mean error for this row
        if squared:
            output[i] = np.mean(row ** 2)
        else:
            output[i] = np.mean(np.abs(row))

    return output


@numba.njit(parallel=False)
def _mean_error_columnwise(
    a_data,
    a_indices,
    a_indptr,
    b_pcs,
    b_rotation,
    n_cols,
    squared
):
    """
    Compute column-wise mean reconstruction error (JIT-compiled).

    Aggregates reconstruction errors across all rows for each column/feature.
    Returns mean error per feature, useful for identifying poorly reconstructed genes.

    :param a_data: Non-zero values from sparse CSR matrix
    :param a_indices: Column indices for non-zero values
    :param a_indptr: Index pointers for row boundaries
    :param b_pcs: Principal component scores (n_observations x n_components)
    :param b_rotation: PCA rotation matrix (n_components x n_features)
    :param n_cols: Number of columns in original matrix
    :param squared: If True, compute MSE; if False, compute MAE
    :return: Array of mean errors, one per column
    """

    n_row = b_pcs.shape[0]
    output = np.zeros(n_cols, dtype=float)

    # Accumulate errors across all rows
    for i in numba.prange(n_row):

        # Get indices of non-zero elements in this row
        _idx_a = a_indices[a_indptr[i]:a_indptr[i + 1]]
        _nnz_a = _idx_a.shape[0]

        # Reconstruct row from PCA
        row = b_pcs[i, :] @ b_rotation

        # Subtract actual sparse values from reconstruction
        if _nnz_a == 0:
            pass
        else:
            row[_idx_a] -= a_data[a_indptr[i]:a_indptr[i + 1]]

        # Accumulate column-wise errors
        if squared:
            output += row ** 2
        else:
            output += np.abs(row)

    # Return mean across rows
    return output / n_row


@numba.njit(parallel=False)
def _sum_on_indices(
    data,
    indices,
    max_index,
    squared=False
):
    """
    Sum sparse data values by their index positions (JIT-compiled).

    Aggregates values that share the same index. Used for summing across one
    dimension of a sparse matrix. For CSR matrices, this sums columns; for CSC
    matrices, this sums rows.

    :param data: Non-zero values from sparse matrix
    :param indices: Index for each value (column indices for CSR, row indices for CSC)
    :param max_index: Maximum index value (determines output array size)
    :param squared: If True, sum squared values instead of raw values
    :return: Array of sums, one per unique index from 0 to max_index-1
    """

    output = np.zeros(
        max_index,
        dtype=data.dtype
    )

    if squared:
        # Accumulate squared values at their index positions
        for i in numba.prange(indices.shape[0]):
            output[indices[i]] += data[i] ** 2

    else:
        # Accumulate raw values at their index positions
        for i in numba.prange(indices.shape[0]):
            output[indices[i]] += data[i]

    return output


@numba.njit(parallel=False)
def _sum_on_indptr(
    data,
    indptr,
    squared=False
):
    """
    Sum sparse data values within index pointer ranges (JIT-compiled).

    Uses the indptr array to identify contiguous segments of data and sums within
    each segment. For CSR matrices, this sums rows; for CSC matrices, this sums columns.

    :param data: Non-zero values from sparse matrix
    :param indptr: Index pointers defining segment boundaries. Has length n_segments + 1,
        where segment i contains data[indptr[i]:indptr[i+1]]
    :param squared: If True, sum squared values instead of raw values
    :return: Array of sums, one per segment (length = indptr.shape[0] - 1)
    """

    output = np.zeros(
        indptr.shape[0] - 1,
        dtype=data.dtype
    )

    if squared:
        # Sum squared values within each pointer range
        for i in numba.prange(output.shape[0]):
            output[i] = np.sum(data[indptr[i]:indptr[i + 1]] ** 2)

    else:
        # Sum raw values within each pointer range
        for i in numba.prange(output.shape[0]):
            output[i] = np.sum(data[indptr[i]:indptr[i + 1]])

    return output


@numba.njit(parallel=False)
def _csr_row_divide(data, indptr, row_normalization_vec):
    """
    Divide each row of CSR matrix by corresponding normalization factor (JIT-compiled).

    Operates in-place on the data array. Each contiguous segment defined by indptr
    (representing one row) is divided by its corresponding factor.

    :param data: Non-zero values from CSR matrix (modified in-place)
    :param indptr: Index pointers for row boundaries
    :param row_normalization_vec: Normalization factor for each row
    """

    for i in numba.prange(indptr.shape[0] - 1):
        data[indptr[i]:indptr[i + 1]] /= row_normalization_vec[i]


@numba.njit(parallel=False)
def _csr_column_divide(data, indices, column_normalization_vec):
    """
    Divide each element by its column's normalization factor (JIT-compiled).

    Operates in-place on the data array. Each value is divided by the factor
    corresponding to its column index.

    :param data: Non-zero values from sparse matrix (modified in-place)
    :param indices: Column index for each value in data
    :param column_normalization_vec: Normalization factor for each column
    """

    for i, idx in enumerate(indices):
        data[i] /= column_normalization_vec[idx]


def _csr_column_nnz(indices, n_col):
    """
    Count number of non-zero elements in each column of CSR matrix.

    :param indices: Column indices from CSR matrix
    :param n_col: Total number of columns
    :return: Array of counts, one per column
    """

    return np.bincount(indices, minlength=n_col)


def _csr_to_csc_indptr(indices, n_col):
    """
    Compute CSC-style index pointers from CSR column indices.

    Creates the indptr array that would be used in a CSC representation,
    indicating where each column's data would start in a column-major layout.

    :param indices: Column indices from CSR matrix
    :param n_col: Total number of columns
    :return: Index pointer array of length n_col + 1
    """

    output = np.zeros(n_col + 1, dtype=int)

    # Compute cumulative sum of column counts
    np.cumsum(
        _csr_column_nnz(indices, n_col),
        out=output[1:]
    )

    return output


@numba.njit(parallel=False)
def _csr_extract_columns(data, col_indices, new_col_indptr):
    """
    Reorder CSR data array to be column-major (JIT-compiled).

    Rearranges the data array so that values are grouped by column instead of row.
    This is the core operation for converting CSR to CSC ordering.

    :param data: Non-zero values from CSR matrix
    :param col_indices: Column index for each value in data
    :param new_col_indptr: Target index pointers for column-major layout
    :return: Reordered data array in column-major order
    """

    output_data = np.zeros_like(data)
    col_indptr_used = np.zeros_like(new_col_indptr)

    # Place each value in its column-major position
    for i in range(data.shape[0]):
        _col = col_indices[i]
        _new_pos = new_col_indptr[_col] + col_indptr_used[_col]
        output_data[_new_pos] = data[i]
        col_indptr_used[_col] += 1

    return output_data
