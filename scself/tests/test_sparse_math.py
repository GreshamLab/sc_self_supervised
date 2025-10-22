"""
Generated with claude code (claude-sonnet-4-5)
Most of these look useless to be honest
"""

import unittest
import numpy as np
import numpy.testing as npt
import scipy.sparse as sps

from scself.sparse.math import (
    is_csr,
    is_csc,
    mcv_mean_error_sparse,
    sparse_sum,
    sparse_normalize_columns,
    sparse_normalize_total,
    sparse_csr_extract_columns,
    _csr_column_nnz,
    _csr_to_csc_indptr
)


class TestFormatChecking(unittest.TestCase):
    """Test sparse format detection functions."""

    def setUp(self):
        self.data = np.random.rand(10, 5)
        self.data[self.data < 0.5] = 0
        return super().setUp()

    def test_is_csr_matrix(self):
        """Test is_csr with csr_matrix (legacy format)."""
        csr = sps.csr_matrix(self.data)
        self.assertTrue(is_csr(csr))
        self.assertFalse(is_csc(csr))

    def test_is_csr_array(self):
        """Test is_csr with csr_array (new format)."""
        csr = sps.csr_array(self.data)
        self.assertTrue(is_csr(csr))
        self.assertFalse(is_csc(csr))

    def test_is_csc_matrix(self):
        """Test is_csc with csc_matrix (legacy format)."""
        csc = sps.csc_matrix(self.data)
        self.assertTrue(is_csc(csc))
        self.assertFalse(is_csr(csc))

    def test_is_csc_array(self):
        """Test is_csc with csc_array (new format)."""
        csc = sps.csc_array(self.data)
        self.assertTrue(is_csc(csc))
        self.assertFalse(is_csr(csc))

    def test_is_csr_dense_array(self):
        """Test that dense arrays are not detected as CSR."""
        self.assertFalse(is_csr(self.data))
        self.assertFalse(is_csc(self.data))

    def test_is_csr_other_sparse(self):
        """Test that other sparse formats are not detected as CSR/CSC."""
        coo = sps.coo_matrix(self.data)
        self.assertFalse(is_csr(coo))
        self.assertFalse(is_csc(coo))


class TestMCVMeanErrorSparse(unittest.TestCase):
    """Test sparse PCA reconstruction error computation."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12345)

        cls.X = rng.random((100, 20))
        cls.X[cls.X < 0.3] = 0  # Make sparse
        cls.PC = rng.random((100, 10))
        cls.ROTATION = rng.random((10, 20))
        cls.Y = cls.PC @ cls.ROTATION

        # Expected values for dense computation
        cls.MSE_scalar = np.mean((cls.X - cls.Y) ** 2)
        cls.MSE_row = np.mean((cls.X - cls.Y) ** 2, axis=1)
        cls.MSE_col = np.mean((cls.X - cls.Y) ** 2, axis=0)

        cls.MAE_scalar = np.mean(np.abs(cls.X - cls.Y))
        cls.MAE_row = np.mean(np.abs(cls.X - cls.Y), axis=1)
        cls.MAE_col = np.mean(np.abs(cls.X - cls.Y), axis=0)

    def test_mse_rowwise(self):
        """Test MSE computation row-wise."""
        X_sparse = sps.csr_matrix(self.X)

        result = mcv_mean_error_sparse(
            X_sparse,
            self.PC,
            self.ROTATION,
            axis=1,
            squared=True
        )

        npt.assert_array_almost_equal(result, self.MSE_row)

    def test_mse_columnwise(self):
        """Test MSE computation column-wise."""
        X_sparse = sps.csr_matrix(self.X)

        result = mcv_mean_error_sparse(
            X_sparse,
            self.PC,
            self.ROTATION,
            axis=0,
            squared=True
        )

        npt.assert_array_almost_equal(result, self.MSE_col)

    def test_mse_scalar(self):
        """Test MSE computation flattened."""
        X_sparse = sps.csr_matrix(self.X)

        result = mcv_mean_error_sparse(
            X_sparse,
            self.PC,
            self.ROTATION,
            axis=None,
            squared=True
        )

        npt.assert_almost_equal(result, self.MSE_scalar)

    def test_mae_rowwise(self):
        """Test MAE computation row-wise."""
        X_sparse = sps.csr_matrix(self.X)

        result = mcv_mean_error_sparse(
            X_sparse,
            self.PC,
            self.ROTATION,
            axis=1,
            squared=False
        )

        npt.assert_array_almost_equal(result, self.MAE_row)

    def test_mae_columnwise(self):
        """Test MAE computation column-wise."""
        X_sparse = sps.csr_matrix(self.X)

        result = mcv_mean_error_sparse(
            X_sparse,
            self.PC,
            self.ROTATION,
            axis=0,
            squared=False
        )

        npt.assert_array_almost_equal(result, self.MAE_col)

    def test_mae_scalar(self):
        """Test MAE computation flattened."""
        X_sparse = sps.csr_matrix(self.X)

        result = mcv_mean_error_sparse(
            X_sparse,
            self.PC,
            self.ROTATION,
            axis=None,
            squared=False
        )

        npt.assert_almost_equal(result, self.MAE_scalar)

    def test_invalid_axis_raises_error(self):
        """Test that invalid axis raises ValueError."""
        X_sparse = sps.csr_matrix(self.X)

        with self.assertRaises(ValueError):
            mcv_mean_error_sparse(
                X_sparse,
                self.PC,
                self.ROTATION,
                axis=2,
                squared=True
            )

    def test_empty_sparse_matrix(self):
        """Test with completely sparse (all zeros) matrix."""
        X_empty = sps.csr_matrix((100, 20))

        result = mcv_mean_error_sparse(
            X_empty,
            self.PC,
            self.ROTATION,
            axis=1,
            squared=True
        )

        # Error should just be the reconstruction
        expected = np.mean(self.Y ** 2, axis=1)
        npt.assert_array_almost_equal(result, expected)


class TestSparseSum(unittest.TestCase):
    """Test sparse summation operations."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.data = rng.random((50, 30))
        self.data[self.data < 0.4] = 0

        self.csr = sps.csr_matrix(self.data)
        self.csc = sps.csc_matrix(self.data)
        return super().setUp()

    def test_sum_all_csr(self):
        """Test summing all elements in CSR matrix."""
        result = sparse_sum(self.csr, axis=None)
        expected = np.sum(self.data)

        npt.assert_almost_equal(result, expected)

    def test_sum_all_csc(self):
        """Test summing all elements in CSC matrix."""
        result = sparse_sum(self.csc, axis=None)
        expected = np.sum(self.data)

        npt.assert_almost_equal(result, expected)

    def test_sum_axis0_csr(self):
        """Test column sums for CSR matrix."""
        result = sparse_sum(self.csr, axis=0)
        expected = np.sum(self.data, axis=0)

        npt.assert_array_almost_equal(result, expected)

    def test_sum_axis0_csc(self):
        """Test column sums for CSC matrix."""
        result = sparse_sum(self.csc, axis=0)
        expected = np.sum(self.data, axis=0)

        npt.assert_array_almost_equal(result, expected)

    def test_sum_axis1_csr(self):
        """Test row sums for CSR matrix."""
        result = sparse_sum(self.csr, axis=1)
        expected = np.sum(self.data, axis=1)

        npt.assert_array_almost_equal(result, expected)

    def test_sum_axis1_csc(self):
        """Test row sums for CSC matrix."""
        result = sparse_sum(self.csc, axis=1)
        expected = np.sum(self.data, axis=1)

        npt.assert_array_almost_equal(result, expected)

    def test_sum_dense_raises_error(self):
        """Test that dense arrays raise ValueError."""
        with self.assertRaises(ValueError) as context:
            sparse_sum(self.data, axis=None)

        self.assertIn('sparse array', str(context.exception))

    def test_sum_invalid_format_raises_error(self):
        """Test that unsupported sparse formats raise ValueError."""
        coo = sps.coo_matrix(self.data)

        with self.assertRaises(ValueError):
            sparse_sum(coo, axis=0)

    def test_sum_empty_matrix(self):
        """Test sum of empty sparse matrix."""
        empty = sps.csr_matrix((10, 5))
        result = sparse_sum(empty, axis=None)

        self.assertEqual(result, 0)

    def test_sum_single_row(self):
        """Test sum with single row matrix."""
        single_row = sps.csr_matrix(self.data[0:1, :])
        result = sparse_sum(single_row, axis=1)

        npt.assert_array_almost_equal(result, [np.sum(self.data[0, :])])


class TestSparseNormalizeColumns(unittest.TestCase):
    """Test column-wise normalization of sparse matrices."""

    def setUp(self):
        rng = np.random.default_rng(100)
        self.data = rng.integers(0, 10, (40, 20)).astype(float)
        self.data[self.data < 3] = 0

        self.col_factors = rng.random(20) + 0.5  # Avoid division by zero
        return super().setUp()

    def test_normalize_columns_csr(self):
        """Test column normalization for CSR matrix."""
        csr = sps.csr_matrix(self.data.copy())
        sparse_normalize_columns(csr, self.col_factors)

        expected = self.data / self.col_factors[None, :]
        npt.assert_array_almost_equal(csr.toarray(), expected)

    def test_normalize_columns_modifies_inplace(self):
        """Test that normalization modifies matrix in-place."""
        csr = sps.csr_matrix(self.data.copy())
        data_ptr_before = id(csr.data)

        sparse_normalize_columns(csr, self.col_factors)
        data_ptr_after = id(csr.data)

        # Should modify same data array
        self.assertEqual(data_ptr_before, data_ptr_after)

    def test_normalize_columns_converts_int_to_float(self):
        """Test that integer data is converted to float."""
        int_data = np.random.randint(0, 10, (10, 5))
        int_data[int_data < 3] = 0
        csr = sps.csr_matrix(int_data)

        # Should be integer initially
        self.assertTrue(np.issubdtype(csr.data.dtype, np.integer))

        sparse_normalize_columns(csr, np.ones(5))

        # Should be float after normalization
        self.assertTrue(np.issubdtype(csr.data.dtype, np.floating))

    def test_normalize_columns_int32_to_float32(self):
        """Test int32 conversion to float32."""
        int_data = np.random.randint(0, 10, (10, 5), dtype=np.int32)
        int_data[int_data < 3] = 0
        csr = sps.csr_matrix(int_data)

        sparse_normalize_columns(csr, np.ones(5))

        self.assertEqual(csr.data.dtype, np.float32)

    def test_normalize_columns_int64_to_float64(self):
        """Test int64 conversion to float64."""
        int_data = np.random.randint(0, 10, (10, 5), dtype=np.int64)
        int_data[int_data < 3] = 0
        csr = sps.csr_matrix(int_data)

        sparse_normalize_columns(csr, np.ones(5))

        self.assertEqual(csr.data.dtype, np.float64)

    def test_normalize_columns_csc_raises_error(self):
        """Test that CSC format raises ValueError."""
        csc = sps.csc_matrix(self.data)

        with self.assertRaises(ValueError) as context:
            sparse_normalize_columns(csc, self.col_factors)

        self.assertIn('csr', str(context.exception).lower())

    def test_normalize_by_ones_unchanged(self):
        """Test that normalizing by ones leaves data unchanged."""
        csr = sps.csr_matrix(self.data.copy())
        expected = csr.toarray().copy()

        sparse_normalize_columns(csr, np.ones(20))

        npt.assert_array_almost_equal(csr.toarray(), expected)


class TestSparseNormalizeTotal(unittest.TestCase):
    """Test total count normalization."""

    def setUp(self):
        rng = np.random.default_rng(200)
        self.data = rng.integers(0, 20, (50, 30)).astype(float)
        self.data[self.data < 5] = 0
        return super().setUp()

    def test_normalize_total_csr_with_target(self):
        """Test normalization with explicit target sum."""
        csr = sps.csr_matrix(self.data.copy())
        target = 100.0

        sparse_normalize_total(csr, target_sum=target)

        row_sums = np.array(csr.sum(axis=1)).flatten()
        # Rows with data should sum to ~target
        non_zero_rows = self.data.sum(axis=1) > 0
        npt.assert_array_almost_equal(
            row_sums[non_zero_rows],
            np.full(non_zero_rows.sum(), target)
        )

    def test_normalize_total_csc_with_target(self):
        """Test normalization with CSC format."""
        csc = sps.csc_matrix(self.data.copy())
        target = 100.0

        sparse_normalize_total(csc, target_sum=target)

        row_sums = np.array(csc.sum(axis=1)).flatten()
        non_zero_rows = self.data.sum(axis=1) > 0
        npt.assert_array_almost_equal(
            row_sums[non_zero_rows],
            np.full(non_zero_rows.sum(), target)
        )

    def test_normalize_total_with_median(self):
        """Test normalization with None target (uses median)."""
        csr = sps.csr_matrix(self.data.copy())

        sparse_normalize_total(csr, target_sum=None)

        row_sums = np.array(csr.sum(axis=1)).flatten()
        original_sums = self.data.sum(axis=1)
        expected_target = np.median(original_sums)

        non_zero_rows = original_sums > 0
        npt.assert_array_almost_equal(
            row_sums[non_zero_rows],
            np.full(non_zero_rows.sum(), expected_target)
        )

    def test_normalize_total_with_size_factors(self):
        """Test normalization with pre-computed size factors."""
        csr = sps.csr_matrix(self.data.copy())
        size_factors = np.ones(50) * 2  # Divide everything by 2

        sparse_normalize_total(csr, size_factor=size_factors)

        expected = self.data / 2
        npt.assert_array_almost_equal(csr.toarray(), expected)

    def test_normalize_total_empty_rows(self):
        """Test that empty rows are handled correctly."""
        data_with_empty = self.data.copy()
        data_with_empty[0, :] = 0  # Empty first row
        data_with_empty[5, :] = 0  # Empty middle row

        csr = sps.csr_matrix(data_with_empty)
        sparse_normalize_total(csr, target_sum=100)

        # Empty rows should remain empty
        npt.assert_array_equal(csr.toarray()[0, :], 0)
        npt.assert_array_equal(csr.toarray()[5, :], 0)

    def test_normalize_total_converts_to_float(self):
        """Test that integer data is converted to float."""
        int_data = np.random.randint(0, 10, (10, 5))
        int_data[int_data < 3] = 0
        csr = sps.csr_matrix(int_data)

        sparse_normalize_total(csr, target_sum=50)

        self.assertTrue(np.issubdtype(csr.data.dtype, np.floating))

    def test_normalize_total_invalid_format_raises_error(self):
        """Test that unsupported formats raise ValueError."""
        coo = sps.coo_matrix(self.data)

        with self.assertRaises(ValueError) as context:
            sparse_normalize_total(coo, target_sum=100)

        self.assertIn('csr_array or csc_array', str(context.exception))

    def test_normalize_total_default_target(self):
        """Test default target_sum value."""
        csr = sps.csr_matrix(self.data.copy())

        # Default target_sum is 10,000 (not None)
        sparse_normalize_total(csr)

        row_sums = np.array(csr.sum(axis=1)).flatten()
        non_zero_rows = self.data.sum(axis=1) > 0

        # With default (10,000), rows should sum to 10,000
        npt.assert_array_almost_equal(
            row_sums[non_zero_rows],
            np.full(non_zero_rows.sum(), 10000.0)
        )


class TestSparseCSRExtractColumns(unittest.TestCase):
    """Test CSR to column-major reordering."""

    def setUp(self):
        rng = np.random.default_rng(300)
        self.data = rng.random((20, 15))
        self.data[self.data < 0.5] = 0
        self.csr = sps.csr_matrix(self.data)
        return super().setUp()

    def test_extract_columns_returns_csc_matrix(self):
        """Test that fake_csc_matrix=True returns CSC matrix."""
        result = sparse_csr_extract_columns(self.csr, fake_csc_matrix=True)

        self.assertIsInstance(result, sps.csc_matrix)
        self.assertEqual(result.shape, self.csr.shape)

    def test_extract_columns_returns_arrays(self):
        """Test that fake_csc_matrix=False returns arrays."""
        data, indptr = sparse_csr_extract_columns(
            self.csr,
            fake_csc_matrix=False
        )

        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(indptr, np.ndarray)
        self.assertEqual(data.shape[0], self.csr.data.shape[0])
        self.assertEqual(indptr.shape[0], self.csr.shape[1] + 1)

    def test_extract_columns_indptr_structure(self):
        """Test that indptr has correct structure."""
        _, indptr = sparse_csr_extract_columns(
            self.csr,
            fake_csc_matrix=False
        )

        # First element should be 0
        self.assertEqual(indptr[0], 0)

        # Last element should equal number of non-zeros
        self.assertEqual(indptr[-1], self.csr.nnz)

        # Should be non-decreasing
        self.assertTrue(np.all(np.diff(indptr) >= 0))

    def test_extract_columns_preserves_data_count(self):
        """Test that all non-zero values are preserved."""
        data, _ = sparse_csr_extract_columns(
            self.csr,
            fake_csc_matrix=False
        )

        # Should have same number of non-zeros
        self.assertEqual(data.shape[0], self.csr.nnz)

        # Sum should be preserved
        npt.assert_almost_equal(np.sum(data), np.sum(self.csr.data))

    def test_extract_columns_column_counts(self):
        """Test that column counts are correct."""
        _, indptr = sparse_csr_extract_columns(
            self.csr,
            fake_csc_matrix=False
        )

        # Compute column counts from indptr
        col_counts = np.diff(indptr)

        # Should match actual column counts
        expected_counts = np.array([
            (self.csr.indices == i).sum()
            for i in range(self.csr.shape[1])
        ])

        npt.assert_array_equal(col_counts, expected_counts)

    def test_extract_columns_empty_matrix(self):
        """Test with empty sparse matrix."""
        empty = sps.csr_matrix((10, 5))
        data, indptr = sparse_csr_extract_columns(
            empty,
            fake_csc_matrix=False
        )

        # Should have no data
        self.assertEqual(data.shape[0], 0)

        # Indptr should be all zeros
        npt.assert_array_equal(indptr, np.zeros(6))


class TestHelperFunctions(unittest.TestCase):
    """Test internal helper functions."""

    def setUp(self):
        rng = np.random.default_rng(400)
        self.data = rng.random((15, 10))
        self.data[self.data < 0.5] = 0
        self.csr = sps.csr_matrix(self.data)
        return super().setUp()

    def test_csr_column_nnz(self):
        """Test counting non-zeros per column."""
        result = _csr_column_nnz(self.csr.indices, self.csr.shape[1])

        # Verify counts
        expected = np.array([
            (self.csr.indices == i).sum()
            for i in range(self.csr.shape[1])
        ])

        npt.assert_array_equal(result, expected)

    def test_csr_column_nnz_empty_columns(self):
        """Test with some empty columns."""
        # Create data with empty columns
        data = np.zeros((10, 5))
        data[:, [0, 2, 4]] = np.random.rand(10, 3)
        data[data < 0.5] = 0

        csr = sps.csr_matrix(data)
        result = _csr_column_nnz(csr.indices, csr.shape[1])

        # Empty columns should have count 0
        self.assertEqual(result[1], 0)
        self.assertEqual(result[3], 0)

    def test_csr_to_csc_indptr(self):
        """Test CSC indptr generation."""
        result = _csr_to_csc_indptr(self.csr.indices, self.csr.shape[1])

        # Should have length n_cols + 1
        self.assertEqual(result.shape[0], self.csr.shape[1] + 1)

        # First element should be 0
        self.assertEqual(result[0], 0)

        # Last element should be number of non-zeros
        self.assertEqual(result[-1], self.csr.nnz)

        # Should be non-decreasing
        self.assertTrue(np.all(np.diff(result) >= 0))

    def test_csr_to_csc_indptr_column_ranges(self):
        """Test that indptr correctly defines column ranges."""
        indptr = _csr_to_csc_indptr(self.csr.indices, self.csr.shape[1])

        # Verify each column's count
        for col in range(self.csr.shape[1]):
            expected_count = (self.csr.indices == col).sum()
            actual_count = indptr[col + 1] - indptr[col]
            self.assertEqual(actual_count, expected_count)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_single_element_matrix(self):
        """Test with 1x1 sparse matrix."""
        data = sps.csr_matrix([[5.0]])

        # Test sum
        result = sparse_sum(data, axis=None)
        self.assertEqual(result, 5.0)

        # Test normalize
        sparse_normalize_total(data, target_sum=10.0)
        npt.assert_almost_equal(data.toarray()[0, 0], 10.0)

    def test_all_zeros_matrix(self):
        """Test with completely sparse (all zeros) matrix."""
        zeros = sps.csr_matrix((10, 10))

        result = sparse_sum(zeros, axis=1)
        npt.assert_array_equal(result, np.zeros(10))

    def test_single_nonzero_per_row(self):
        """Test with exactly one non-zero per row."""
        data = np.eye(10)
        csr = sps.csr_matrix(data)

        sparse_normalize_total(csr, target_sum=5.0)

        # Each row should sum to 5
        row_sums = np.array(csr.sum(axis=1)).flatten()
        npt.assert_array_almost_equal(row_sums, np.full(10, 5.0))

    def test_very_sparse_matrix(self):
        """Test with very sparse matrix (< 1% density)."""
        rng = np.random.default_rng(500)
        data = rng.random((1000, 500))
        data[data < 0.99] = 0  # 1% density

        csr = sps.csr_matrix(data)

        # Should still compute correctly
        result = sparse_sum(csr, axis=1)
        expected = data.sum(axis=1)

        npt.assert_array_almost_equal(result, expected)

    def test_dense_sparse_matrix(self):
        """Test with very dense sparse matrix (> 90% density)."""
        rng = np.random.default_rng(600)
        data = rng.random((50, 30))
        data[data < 0.1] = 0  # 90% density

        csr = sps.csr_matrix(data)

        result = sparse_sum(csr, axis=0)
        expected = data.sum(axis=0)

        npt.assert_array_almost_equal(result, expected)

    def test_single_row_matrix(self):
        """Test with single row matrix."""
        data = np.random.rand(1, 20)
        data[data < 0.5] = 0
        csr = sps.csr_matrix(data)

        result = sparse_sum(csr, axis=1)
        self.assertEqual(result.shape[0], 1)

        sparse_normalize_total(csr, target_sum=100)
        npt.assert_almost_equal(csr.sum(), 100)

    def test_single_column_matrix(self):
        """Test with single column matrix."""
        data = np.random.rand(20, 1)
        data[data < 0.5] = 0
        csr = sps.csr_matrix(data)

        result = sparse_sum(csr, axis=0)
        self.assertEqual(result.shape[0], 1)


class TestDataTypeConsistency(unittest.TestCase):
    """Test data type handling across operations."""

    def test_float32_preserved(self):
        """Test that float32 dtype is preserved."""
        data = np.random.rand(10, 5).astype(np.float32)
        data[data < 0.5] = 0
        csr = sps.csr_matrix(data)

        sparse_normalize_total(csr, target_sum=100)

        self.assertEqual(csr.data.dtype, np.float32)

    def test_float64_preserved(self):
        """Test that float64 dtype is preserved."""
        data = np.random.rand(10, 5).astype(np.float64)
        data[data < 0.5] = 0
        csr = sps.csr_matrix(data)

        sparse_normalize_total(csr, target_sum=100)

        self.assertEqual(csr.data.dtype, np.float64)

    def test_sum_dtype_matches_input(self):
        """Test that sparse_sum returns dtype matching input."""
        data32 = np.random.rand(10, 5).astype(np.float32)
        data32[data32 < 0.5] = 0
        csr32 = sps.csr_matrix(data32)

        result = sparse_sum(csr32, axis=1)
        self.assertEqual(result.dtype, np.float32)

        data64 = data32.astype(np.float64)
        csr64 = sps.csr_matrix(data64)

        result = sparse_sum(csr64, axis=1)
        self.assertEqual(result.dtype, np.float64)
