import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal

    backend = "numpy"
import numpy as npp
import pytest
from mpi4py import MPI

from pylops.basicoperators import FirstDerivative, Identity, MatrixMult
from pylops.optimization.callback import CostToInitialCallback
from pylops.optimization.cls_sparsity import IRLS
from pylops.optimization.sparsity import fista, irls, ista, omp, spgl1, splitbregman

from pylops_mpi.DistributedArray import DistributedArray
from pylops_mpi.basicoperators.BlockDiag import MPIBlockDiag

par1 = {
    "ny": 11,
    "nx": 11,
    "imag": 0,
    "x0": False,
    "dtype": "float64",
}  # square real, zero initial guess
par2 = {
    "ny": 31,
    "nx": 11,
    "imag": 0,
    "x0": False,
    "dtype": "float64",
}  # overdetermined real, zero initial guess
par3 = {
    "ny": 21,
    "nx": 41,
    "imag": 0,
    "x0": True,
    "dtype": "float64",
}  # underdetermined real, non-zero initial guess
par1j = {
    "ny": 11,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex128",
}  # square complex, zero initial guess
par2j = {
    "ny": 31,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex128",
}  # overdetermined complex, zero initial guess
par3j = {
    "ny": 21,
    "nx": 41,
    "imag": 1j,
    "x0": True,
    "dtype": "complex128",
}  # underdetermined complex, non-zero initial guess

size = MPI.COMM_WORLD.Get_size()

@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par1j), (par2j), (par3j)])
def test_ISTA_FISTA(par):
    """Invert problem with ISTA/FISTA"""
    npp.random.seed(42)
    A = npp.random.randn(par["ny"], par["nx"]) + par["imag"] * npp.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MPIBlockDiag(ops=[MatrixMult(np.asarray(A), dtype=par["dtype"])])

    x = np.random.rand(size * par["nx"]) + par["imag"] * np.random.rand(size * par["nx"])
    # x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    # x[3] = 1.0 + par["imag"] * 1.0
    # x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    x = DistributedArray.to_dist(x)
    y = Aop * x

    eps = 1.0 if par["ny"] >= par["nx"] else 2.0
    maxit = 500

    # Regularization based ISTA and FISTA
    threshkinds = ["hard", "soft", "half"]
    for threshkind in threshkinds:
        for preallocate in [False, True]:
            xinv1, _, _ = ista(
                Aop,
                y,
                niter=maxit,
                eps=eps,
                threshkind=threshkind,
                tol=0,
                preallocate=preallocate,
            )
