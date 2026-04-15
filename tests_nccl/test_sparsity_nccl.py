"""Test Sparsity solvers
    Designed to run with n GPUs (with 1 MPI process per GPU)
    $ mpiexec -n 10 pytest test_sparsity_nccl.py --with-mpi

This file employs the same test sets as test_sparsity under NCCL environment
"""
import pytest
from mpi4py import MPI
from numpy.testing import assert_allclose
import numpy as np
import cupy as cp


from pylops.basicoperators import BlockDiag, MatrixMult, Identity, HStack, VStack
from pylops import ista as ista_pylops, fista as fista_pylops

from pylops_mpi.basicoperators import MPIBlockDiag, MPIVStack, MPIHStack
from pylops_mpi.DistributedArray import DistributedArray, Partition
from pylops_mpi.optimization.sparsity import ista, fista
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


def test_ISTA_FISTA_unknown_threshkind():
    """Check error is raised if unknown threshkind is passed"""
    with pytest.raises(ValueError, match="threshkind must be"):
        y = DistributedArray(global_shape=size * 5, engine="cupy", base_comm_nccl=nccl_comm)
        y[:] = 1
        _ = ista(MPIBlockDiag(ops=[Identity(5)]), y, 10, threshkind="foo")
    with pytest.raises(ValueError, match="threshkind must be"):
        y = DistributedArray(global_shape=size * 5, engine="cupy", base_comm_nccl=nccl_comm)
        y[:] = 1
        _ = fista(MPIBlockDiag(ops=[Identity(5)]), y, 10, threshkind="foo")


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par1j), (par2j), (par3j)])
def test_ISTA_FISTA_alpha_too_high(par):
    """Invert problem with ISTA/FISTA NCCL- alpha too high"""
    A = cp.random.randn(par["ny"], par["nx"]) + par["imag"] * cp.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MPIBlockDiag(ops=[MatrixMult(cp.asarray(A), dtype=par["dtype"])], dtype=par['dtype'])

    x = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    x[:] = cp.zeros(par["nx"], dtype=par['dtype']) + par["imag"] * cp.zeros(par["nx"], dtype=par['dtype'])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x

    if par["x0"]:
        x0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'], engine="cupy")
        x0[:] = cp.ones(par["nx"], dtype=par['dtype']) + par["imag"] * cp.ones(par["nx"], dtype=par['dtype'])
    else:
        # Set TO 0s if x0 = False
        x0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'], engine="cupy")
        x0[:] = 0

    # Check that exception is raised
    for solver in [ista, fista]:
        with pytest.raises(ValueError, match="due to residual increasing"):
            xinv, _, _ = solver(
                Aop,
                y,
                x0,
                niter=100,
                eps=0.1,
                alpha=1e5,
                monitorres=True,
                tol=0
            )
        _, _, cost = solver(
            Aop,
            y,
            x0,
            niter=100,
            eps=0.1,
            alpha=1e5,
            tol=0,
        )
        assert cp.isnan(cost[-1]) or cp.isinf(cost[-1])


@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_ISTA_FISTA(par):
    """Invert problem with ISTA/FISTA NCCL"""
    d = cp.linspace(1, 10, min(par["ny"], par["nx"]))
    A = cp.zeros((par["ny"], par["nx"]), dtype=par["dtype"])
    cp.fill_diagonal(A, d)
    A = A + par["imag"] * A
    Aop = MPIBlockDiag(ops=[MatrixMult(cp.asarray(A), dtype=par["dtype"])], dtype=par['dtype'])

    x = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    x[:] = cp.zeros(par["nx"], dtype=par['dtype']) + par["imag"] * cp.zeros(par["nx"], dtype=par['dtype'])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x
    x_array = x.asarray()

    if par["x0"]:
        x0 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
        x0[:] = cp.ones(par["nx"], dtype=par['dtype']) + par["imag"] * cp.ones(par["nx"], dtype=par['dtype'])
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        x0 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
        x0[:] = 0
        x0_global = x0.asarray()

    eps = 1.0
    maxit = 200

    if rank == 0:
        d = np.linspace(1, 10, min(par["ny"], par["nx"]))
        A = np.zeros((par["ny"], par["nx"]), dtype=par["dtype"])
        np.fill_diagonal(A, d)
        A = A + par["imag"] * A
        mats = [A.copy() for _ in range(size)]
        ops = [MatrixMult(mats[i], dtype=par["dtype"]) for i in range(size)]
        Aop1 = BlockDiag(ops=ops, forceflat=True)
        y1 = Aop1 * x_array.get()

    # Regularization based ISTA
    threshkinds = ["soft", "half"]
    for threshkind in threshkinds:
        for solver in [ista, fista]:
            xinv, _, _ = solver(
                Aop,
                y,
                x0,
                niter=maxit,
                eps=eps,
                threshkind=threshkind,
                tol=0,
                show=True
            )
            xinv_array = xinv.asarray()
            if rank == 0:
                solver_pylops = ista_pylops if solver is ista else fista_pylops
                xinv1, _, _ = solver_pylops(
                    Aop1,
                    y1,
                    x0=x0_global.get(),
                    niter=maxit,
                    eps=eps,
                    threshkind=threshkind,
                    tol=0,
                    show=True
                )
                assert_allclose(xinv_array.get(), xinv1, rtol=1e-8)


@pytest.mark.parametrize("par", [(par2), (par2j)])
def test_ISTA_FISTA_broadcastmodel(par):
    """Invert problem with ISTA/FISTA NCCL with broadcasted model"""
    d = cp.linspace(1, 10, min(par["ny"], par["nx"]))
    A = cp.zeros((par["ny"], par["nx"]), dtype=par["dtype"])
    cp.fill_diagonal(A, d)
    A = A + par["imag"] * A
    Aop = MPIVStack(ops=[MatrixMult(cp.asarray(A), dtype=par["dtype"])], dtype=par['dtype'])

    x = DistributedArray(global_shape=par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'],
                         partition=Partition.BROADCAST, engine="cupy")
    x[:] = cp.zeros(par["nx"], dtype=par['dtype']) + par["imag"] * cp.zeros(par["nx"], dtype=par['dtype'])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x
    x_array = x.asarray()

    if par["x0"]:
        x0 = DistributedArray(global_shape=par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'],
                              partition=Partition.BROADCAST, engine="cupy")
        x0[:] = cp.ones(par["nx"], dtype=par['dtype']) + par["imag"] * cp.ones(par["nx"], dtype=par['dtype'])
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        x0 = DistributedArray(global_shape=par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'],
                              partition=Partition.BROADCAST, engine="cupy")
        x0[:] = 0
        x0_global = x0.asarray()

    eps = 1.0
    maxit = 200

    if rank == 0:
        d = np.linspace(1, 10, min(par["ny"], par["nx"]))
        A = np.zeros((par["ny"], par["nx"]), dtype=par["dtype"])
        np.fill_diagonal(A, d)
        A = A + par["imag"] * A
        mats = [A.copy() for _ in range(size)]
        ops = [MatrixMult(mats[i], dtype=par["dtype"]) for i in range(size)]
        Aop1 = VStack(ops=ops, forceflat=True)
        y1 = Aop1 * x_array.get()

    # Regularization based ISTA
    threshkinds = ["soft", "half"]
    for threshkind in threshkinds:
        for solver in [ista, fista]:
            xinv, _, _ = solver(
                Aop,
                y,
                x0,
                niter=maxit,
                eps=eps,
                threshkind=threshkind,
                tol=0,
            )
            xinv_array = xinv.asarray()
            if rank == 0:
                solver_pylops = ista_pylops if solver is ista else fista_pylops
                xinv1, _, _ = solver_pylops(
                    Aop1,
                    y1,
                    x0=x0_global.get(),
                    niter=maxit,
                    eps=eps,
                    threshkind=threshkind,
                    tol=0,
                )
                assert_allclose(xinv_array.get(), xinv1, rtol=1e-8)


@pytest.mark.parametrize("par", [(par3), (par3j)])
def test_ISTA_FISTA_broadcastdata(par):
    """Invert problem with ISTA/FISTA NCCL with broadcasted data"""
    A = (rank + 1) * cp.ones((par["ny"], par["nx"])) + (rank + 2) * par[
        "imag"
    ] * cp.ones((par["ny"], par["nx"]))
    Aop = MPIHStack(ops=[MatrixMult(cp.asarray(A), dtype=par["dtype"])], dtype=par['dtype'])

    x = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    x[:] = cp.zeros(par["nx"], dtype=par['dtype']) + par["imag"] * cp.zeros(par["nx"], dtype=par['dtype'])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x
    x_array = x.asarray()

    if par["x0"]:
        x0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'], engine="cupy")
        x0[:] = cp.ones(par["nx"], dtype=par['dtype']) + par["imag"] * cp.ones(par["nx"], dtype=par['dtype'])
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        x0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'], engine="cupy")
        x0[:] = 0
        x0_global = x0.asarray()

    eps = 1.0
    maxit = 200

    if rank == 0:
        ops = [MatrixMult((i + 1) * np.ones((par["ny"], par["nx"])) + (i + 2) * par[
            "imag"
        ] * np.ones((par["ny"], par["nx"])), dtype=par['dtype']) for i in range(size)]
        Aop1 = HStack(ops=ops, forceflat=True)
        y1 = Aop1 * x_array.get()

    # Regularization based ISTA
    threshkinds = ["soft", "half"]
    for threshkind in threshkinds:
        for solver in [ista, fista]:
            xinv, _, _ = solver(
                Aop,
                y,
                x0,
                niter=maxit,
                eps=eps,
                threshkind=threshkind,
                tol=0,
            )
            xinv_array = xinv.asarray()
            if rank == 0:
                solver_pylops = ista_pylops if solver is ista else fista_pylops
                xinv1, _, _ = solver_pylops(
                    Aop1,
                    y1,
                    x0=x0_global.get(),
                    niter=maxit,
                    eps=eps,
                    threshkind=threshkind,
                    tol=0,
                )
                assert_allclose(xinv_array.get(), xinv1, rtol=1e-8)
