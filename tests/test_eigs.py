import os
import pytest

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    backend = "cupy"
else:
    import numpy as np
    backend = "numpy"
import numpy as npp
from pylops.basicoperators import MatrixMult
from pylops.utils.backend import to_numpy

from pylops_mpi.DistributedArray import DistributedArray, StackedDistributedArray
from pylops_mpi.basicoperators.BlockDiag import MPIBlockDiag, MPIStackedBlockDiag
from pylops_mpi.optimization.eigs import power_iteration

par1 = {"n": 21, "imag": 0, "dtype": "float64"}  # square, real
par2 = {"n": 21, "imag": 1j, "dtype": "complex128"}  # square, complex


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_power_iteration(par):
    """Max eigenvalue computation with power iteration method vs. numpy.linalg.eig"""
    np.random.seed(10)

    A = np.random.randn(par["n"], par["n"]) + par["imag"] * np.random.randn(
        par["n"], par["n"]
    )
    A1 = np.conj(A.T) @ A

    # non-symmetric
    Aop = MPIBlockDiag(ops=[MatrixMult(A)], dtype=par['dtype'])
    b_k = DistributedArray(global_shape=Aop.shape[1], dtype=par['dtype'], engine=backend)
    eig = power_iteration(Aop, b_k=b_k, niter=200, tol=0, backend=backend, dtype=par['dtype'])[0]
    eig_np = npp.max(npp.abs(npp.linalg.eig(to_numpy(A))[0]))

    assert np.abs(np.abs(eig) - eig_np) < 1e-3

    # symmetric
    A1op = MPIBlockDiag(ops=[MatrixMult(A1)], dtype=par['dtype'])
    b1_k = DistributedArray(global_shape=A1op.shape[1], dtype=par['dtype'], engine=backend)
    eig = power_iteration(A1op, b_k=b1_k, niter=200, tol=0, backend=backend, dtype=par['dtype'])[0]
    eig_np = npp.max(npp.abs(npp.linalg.eig(to_numpy(A1))[0]))

    assert np.abs(np.abs(eig) - eig_np) < 1e-3


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_power_iteration_stacked(par):
    """Max eigenvalue computation with power iteration method vs. numpy.linalg.eig for MPIStackedBlockDiag"""
    np.random.seed(10)

    A = np.random.randn(par["n"], par["n"]) + par["imag"] * np.random.randn(
        par["n"], par["n"]
    )
    A1 = np.conj(A.T) @ A

    # non-symmetric
    BDiag = MPIBlockDiag(ops=[MatrixMult(A)], dtype=par['dtype'])
    Aop = MPIStackedBlockDiag(ops=[BDiag, BDiag], dtype=par['dtype'])
    dist1 = DistributedArray(global_shape=BDiag.shape[1], dtype=par['dtype'], engine=backend)
    b_k = StackedDistributedArray(distarrays=[dist1, dist1])
    eig = power_iteration(Aop, b_k=b_k, niter=200, tol=0, backend=backend, dtype=par['dtype'])[0]
    eig_np = npp.max(npp.abs(npp.linalg.eig(to_numpy(A))[0]))

    assert np.abs(np.abs(eig) - eig_np) < 1e-3

    # symmetric
    BDiag1 = MPIBlockDiag(ops=[MatrixMult(A1)], dtype=par['dtype'])
    A1op = MPIStackedBlockDiag(ops=[BDiag1, BDiag1], dtype=par['dtype'])
    dist1 = DistributedArray(global_shape=BDiag.shape[1], dtype=par['dtype'], engine=backend)
    b1_k = StackedDistributedArray(distarrays=[dist1, dist1])
    eig = power_iteration(A1op, b_k=b1_k, niter=200, tol=0, backend=backend, dtype=par['dtype'])[0]
    eig_np = npp.max(npp.abs(npp.linalg.eig(to_numpy(A1))[0]))

    assert np.abs(np.abs(eig) - eig_np) < 1e-3
