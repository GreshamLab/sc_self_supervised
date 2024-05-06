import scipy.sparse as sps


def is_csr(x):
    return sps.isspmatrix_csr(x) or isinstance(x, sps.csr_array)


def is_csc(x):
    return sps.isspmatrix_csc(x) or isinstance(x, sps.csc_array)
