import types
import scipy.sparse as sps

try:
    from sparse_dot_mkl import dot_product_mkl as dot

    def sparse_dot_patch(spmat):

        spmat.dot = types.MethodType(dot, spmat)

except ImportError as err:

    import warnings

    warnings.warn(
        "Unable to use MKL for sparse matrix math, "
        "defaulting to numpy/scipy matmul: "
        f"{str(err)}"
    )

    def dot(x, y, dense=False, cast=False, out=None):

        z = x @ y

        if dense and sps.issparse(z):
            z = z.A

        if out is not None:
            out[:] = z
            return out
        else:
            return z

    def sparse_dot_patch(spmat):

        pass
