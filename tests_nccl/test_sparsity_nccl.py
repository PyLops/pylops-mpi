"""Test Sparsity solvers
    Designed to run with n GPUs (with 1 MPI process per GPU)
    $ mpiexec -n 10 pytest test_sparsity_nccl.py --with-mpi

This file employs the same test sets as test_sparsity_nccl under NCCL environment
"""
import pytest
from mpi4py import MPI
from numpy.testing import assert_allclose
import cupy as cp

from pylops.basicoperators import BlockDiag, MatrixMult, Identity
from pylops import ista as ista_pylops
from pylops.utils import to_numpy

from pylops_mpi.basicoperators import MPIBlockDiag
from pylops_mpi.DistributedArray import DistributedArray
from pylops_mpi.optimization.sparsity import ista
from pylops_mpi.utils._nccl import initialize_nccl_comm

nccl_comm = initialize_nccl_comm()

par1 = {
    "ny": 11,
    "nx": 11,
    "imag": 0,
    "x0": False,
    "dtype": "float64",
}  # square real, zero initial guess
par2 = {
    "ny": 21,
    "nx": 11,
    "imag": 0,
    "x0": False,
    "dtype": "float64",
}  # overdetermined real, zero initial guess
par3 = {
    "ny": 11,
    "nx": 21,
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
    "ny": 21,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex128",
}  # overdetermined complex, zero initial guess
par3j = {
    "ny": 11,
    "nx": 21,
    "imag": 1j,
    "x0": True,
    "dtype": "complex128",
}  # underdetermined complex, non-zero initial guess

cp.random.seed(10)
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

def test_ISTA_unknown_threshkind():
    """Check error is raised if unknown threshkind is passed"""
    with pytest.raises(ValueError, match="threshkind must be"):
        y = DistributedArray(global_shape=size * 5, engine="cupy", base_comm_nccl=nccl_comm)
        y[:] = 1
        _ = ista(MPIBlockDiag(ops=[Identity(5)]), y, 10, threshkind="foo")


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par1j), (par2j), (par3j)])
def test_ISTA_alpha_too_high(par):
    """Invert problem with ISTA NCCL- alpha too high"""
    A = cp.random.randn(par["ny"], par["nx"]) + par["imag"] * cp.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MPIBlockDiag(ops=[MatrixMult(cp.asarray(A), dtype=par["dtype"])], dtype=par['dtype'])

    x = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    x[:] = cp.zeros(par["nx"]) + par["imag"] * cp.zeros(par["nx"])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x

    # Check that exception is raised
    with pytest.raises(ValueError, match="due to residual increasing"):
        xinv, _, _ = ista(
            Aop,
            y,
            niter=100,
            eps=0.1,
            alpha=1e5,
            monitorres=True,
            tol=0
        )
    _, _, cost = ista(
        Aop,
        y,
        niter=100,
        eps=0.1,
        alpha=1e5,
        tol=0,
    )
    assert cp.isinf(cost[-1])


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par1j), (par2j), (par3j)])
def test_ISTA(par):
    """Invert problem with ISTA NCCL"""
    A = cp.random.randn(par["ny"], par["nx"]) + par["imag"] * cp.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MPIBlockDiag(ops=[MatrixMult(cp.asarray(A), dtype=par["dtype"])], dtype=par['dtype'])

    x = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    x[:] = cp.zeros(par["nx"]) + par["imag"] * cp.zeros(par["nx"])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x
    x_array = x.asarray()

    eps = 1.0 if par["ny"] >= par["nx"] else 2.0
    maxit = 1000

    if rank == 0:
        Aop1 = BlockDiag(ops=[MatrixMult(to_numpy(A), dtype=par["dtype"]) for _ in range(size)], forceflat=True)
        y1 = Aop1 * x_array.get()

    # Regularization based ISTA
    threshkinds = ["soft", "half"]
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
                assert_allclose(xinv_array.get(), xinv1, rtol=1e-3)

@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par1j), (par2j), (par3j)])
def test_ISTA_stopping(par):
    """Invert problem with ISTA NCCL- Stopping"""
    A = cp.random.randn(par["ny"], par["nx"]) + par["imag"] * cp.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MPIBlockDiag(ops=[MatrixMult(cp.asarray(A), dtype=par["dtype"])], dtype=par['dtype'])

    x = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    x[:] = cp.zeros(par["nx"]) + par["imag"] * cp.zeros(par["nx"])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x

    rtol = 5e-1

    # Regularization based ISTA
    threshkinds = ["soft", "half"]
    for threshkind in threshkinds:
        for preallocate in [False, True]:
            _, _, cost = ista(
                Aop,
                y,
                niter=500,
                eps=0.5,
                threshkind=threshkind,
                tol=0.0,
                rtol=rtol,
                preallocate=preallocate,
            )
            assert cost[-2] / cost[0] >= rtol
            assert cost[-1] / cost[0] < rtol
