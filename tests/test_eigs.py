import os
import pytest

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_allclose

    backend = "cupy"
else:
    import numpy as np
    backend = "numpy"
import numpy as npp
from pylops.basicoperators import MatrixMult
from pylops.utils.backend import to_numpy

from pylops_mpi.basicoperators.BlockDiag import MPIBlockDiag
from pylops_mpi.optimization.eigs import power_iteration

par1 = {"n": 21, "imag": 0, "dtype": "float64"}  # square, real
par2 = {"n": 21, "imag": 1j, "dtype": "complex128"}  # square, complex

@pytest.mark.parametrize("par", [(par1), (par2)])
def test_power_iteration(par):
    """Max eigenvalue computation with power iteration method vs. scipy methods"""
    np.random.seed(10)

    A = np.random.randn(par["n"], par["n"]) + par["imag"] * np.random.randn(
        par["n"], par["n"]
    )
    A1 = np.conj(A.T) @ A

    # non-symmetric
    Aop = MPIBlockDiag(ops=[MatrixMult(A)], dtype=par['dtype'])
    eig = power_iteration(Aop, niter=200, tol=0, backend=backend, dtype=par['dtype'])[0]
    eig_np = npp.max(npp.abs(npp.linalg.eig(to_numpy(A))[0]))

    assert np.abs(np.abs(eig) - eig_np) < 1e-3

    # symmetric
    A1op = MPIBlockDiag(ops=[MatrixMult(A1)], dtype=par['dtype'])
    eig = power_iteration(A1op, niter=200, tol=0, backend=backend, dtype=par['dtype'])[0]
    eig_np = npp.max(npp.abs(npp.linalg.eig(to_numpy(A1))[0]))

    assert np.abs(np.abs(eig) - eig_np) < 1e-3