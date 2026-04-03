import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_allclose

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_allclose

    backend = "numpy"
import pytest
from mpi4py import MPI

from pylops.basicoperators import MatrixMult, BlockDiag
from pylops.optimization.sparsity import ista as ista_pylops

from pylops_mpi.DistributedArray import DistributedArray
from pylops_mpi.basicoperators.BlockDiag import MPIBlockDiag
from pylops_mpi.optimization.sparsity import ista

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
rank = MPI.COMM_WORLD.Get_rank()
if backend == "cupy":
    device_id = rank % np.cuda.runtime.getDeviceCount()
    np.cuda.Device(device_id).use()

@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par1j), (par2j), (par3j)])
def test_ISTA(par):
    """Invert problem with ISTA"""
    np.random.seed(42)
    A = np.random.randn(par["ny"], par["nx"]) + par["imag"] * np.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MPIBlockDiag(ops=[MatrixMult(np.asarray(A), dtype=par["dtype"])], dtype=par['dtype'])

    x = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'], engine=backend)
    x[:] = np.zeros(par["nx"]) + par["imag"] * np.zeros(par["nx"])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x
    x_arr = x.asarray()
    eps = 1.0 if par["ny"] >= par["nx"] else 2.0
    maxit = 1000

    Aop1 = BlockDiag(ops=[MatrixMult(np.asarray(A), dtype=par["dtype"]) for _ in range(size)])
    y1 = Aop1 * x.asarray()

    # Regularization based ISTA
    threshkinds = ["hard", "soft", "half"]
    for threshkind in threshkinds:
        for preallocate in [False, True]:
            xinv, _, _ = ista(
                Aop,
                y,
                niter=maxit,
                eps=eps,
                threshkind=threshkind,
                tol=0,
                preallocate=preallocate,
            )
            xinv_array = xinv.asarray()
            if rank == 0:
                xinv1, _, _ = ista_pylops(
                    Aop1,
                    y1,
                    niter=maxit,
                    eps=eps,
                    threshkind=threshkind,
                    tol=0,
                    preallocate=preallocate,
                )
                assert_allclose(xinv_array, xinv1, rtol=1e-3)
