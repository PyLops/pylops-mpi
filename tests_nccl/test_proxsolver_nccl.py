"""Test proximal solvers
    Designed to run with n GPUs (with 1 MPI process per GPU)
    $ mpiexec -n 10 pytest test_proxsolver_nccl.py --with-mpi
"""
import cupy as cp
import numpy as np
import pylops
import pytest
from mpi4py import MPI
from numpy.testing import assert_allclose
from pylops import BlockDiag, MatrixMult
from pyproximal import L1, L2
from pyproximal.optimization.primal import ADMML2, ProximalGradient
from pyproximal.optimization.primaldual import PrimalDual

from pylops_mpi import DistributedArray, Partition
from pylops_mpi.basicoperators import MPIBlockDiag, MPIVStack
from pylops_mpi.proximal import MPIL2, MPIProxOperator
from pylops_mpi.proximal.optimization.primal import ADMML2 as MPIADMML2
from pylops_mpi.proximal.optimization.primal import (
    ProximalGradient as MPIProximalGradient,
)
from pylops_mpi.proximal.optimization.primaldual import PrimalDual as MPIPrimalDual
from pylops_mpi.utils._nccl import initialize_nccl_comm

nccl_comm = initialize_nccl_comm()
base_comm = MPI.COMM_WORLD
rank = base_comm.Get_rank()
size = base_comm.Get_size()


par1 = {
    "ny": 11,
    "nx": 11,
    "imag": 0,
    "x0": False,
    "dtype": "float64",
}  # square real, zero initial guess
par2 = {
    "ny": 11,
    "nx": 11,
    "imag": 0,
    "x0": True,
    "dtype": "float64",
}  # square real, non-zero initial guess
par3 = {
    "ny": 31,
    "nx": 11,
    "imag": 0,
    "x0": False,
    "dtype": "float64",
}  # overdetermined real, zero initial guess
par4 = {
    "ny": 31,
    "nx": 11,
    "imag": 0,
    "x0": True,
    "dtype": "float64",
}  # overdetermined real, non-zero initial guess
par1j = {
    "ny": 11,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex128",
}  # square complex, zero initial guess
par2j = {
    "ny": 11,
    "nx": 11,
    "imag": 1j,
    "x0": True,
    "dtype": "complex128",
}  # square complex, non-zero initial guess
par3j = {
    "ny": 31,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex128",
}  # overdetermined complex, zero initial guess
par4j = {
    "ny": 31,
    "nx": 11,
    "imag": 1j,
    "x0": True,
    "dtype": "complex128",
}  # overdetermined complex, non-zero initial guess


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par2), (par2j), (par3), (par3j), (par4), (par4j)]
)
def test_proximalgradient_broadcast(par):
    """ProximalGradient with broabcasted model"""
    cp.random.seed(rank)

    A = cp.random.normal(0, 1, (par["ny"], par["nx"])) + par["imag"] * cp.random.normal(
        0, 1, (par["ny"], par["nx"])
    )
    AVStack_MPI = MPIVStack(
        ops=[
            pylops.MatrixMult(A, dtype=par["dtype"]),
        ]
    )

    x = DistributedArray(
        global_shape=par["nx"],
        dtype=par["dtype"],
        partition=Partition.BROADCAST,
        engine="cupy",
    )
    x[:] = cp.random.normal(1, 10, par["nx"]) + par["imag"] * cp.random.normal(
        10, 10, par["nx"]
    )
    x_global = x.asarray()
    if par["x0"]:
        x0 = DistributedArray(
            global_shape=par["nx"],
            dtype=par["dtype"],
            partition=Partition.BROADCAST,
            engine="cupy",
        )
        x0[:] = cp.random.normal(1, 10, par["nx"]) + par["imag"] * cp.random.normal(
            10, 10, par["nx"]
        )
        x0_global = x0.asarray()
    else:
        # Set to 0s if x0 = False
        x0 = DistributedArray(
            global_shape=par["nx"],
            dtype=par["dtype"],
            partition=Partition.BROADCAST,
            engine="cupy",
        )
        x0[:] = 0
        x0_global = x0.asarray()

    y = AVStack_MPI * x

    # L2 prox
    l2d = MPIL2(Op=AVStack_MPI, b=y, x0=x0)

    # L1 prox
    l1 = L1(sigma=1e-1)
    l1d = MPIProxOperator(l1)

    xinv = MPIProximalGradient(l2d, l1d, x0=x0, tau=1e-3, niter=50, show=True)
    assert isinstance(xinv, DistributedArray)
    xinv_array = xinv.asarray()

    As = cp.vstack(base_comm.allgather(A))
    if rank == 0:
        AVStack = MatrixMult(As, dtype=par["dtype"])
        if par["x0"]:
            x0 = x0_global
        else:
            x0 = cp.zeros(par["nx"], dtype=par["dtype"])
        y1 = AVStack * x_global

        l2local = L2(Op=AVStack, b=y1, x0=x0)
        l1local = L1(sigma=1e-1)

        xinv1 = ProximalGradient(
            l2local, l1local, x0=x0, tau=1e-3, niter=50, show=False
        )
        assert_allclose(xinv_array.get(), xinv1.get(), rtol=1e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par2), (par2j), (par3), (par3j), (par4), (par4j)]
)
def test_proximalgradient_scatter(par):
    """ProximalGradient with scattered model"""
    cp.random.seed(rank)

    A = cp.random.normal(0, 1, (par["ny"], par["nx"])) + par["imag"] * cp.random.normal(
        0, 1, (par["ny"], par["nx"])
    )
    ABDiag_MPI = MPIBlockDiag(
        ops=[
            pylops.MatrixMult(A, dtype=par["dtype"]),
        ]
    )

    x = DistributedArray(
        global_shape=par["nx"] * size,
        dtype=par["dtype"],
        partition=Partition.SCATTER,
        engine="cupy",
    )
    x[:] = cp.random.normal(1, 10, par["nx"]) + par["imag"] * cp.random.normal(
        10, 10, par["nx"]
    )
    x_global = x.asarray()
    if par["x0"]:
        x0 = DistributedArray(
            global_shape=par["nx"] * size,
            dtype=par["dtype"],
            partition=Partition.SCATTER,
            engine="cupy",
        )
        x0[:] = cp.random.normal(1, 10, par["nx"]) + par["imag"] * cp.random.normal(
            10, 10, par["nx"]
        )
        x0_global = x0.asarray()
    else:
        # Set to 0s if x0 = False
        x0 = DistributedArray(
            global_shape=par["nx"] * size,
            dtype=par["dtype"],
            partition=Partition.SCATTER,
            engine="cupy",
        )
        x0[:] = 0
        x0_global = x0.asarray()

    y = ABDiag_MPI * x

    # L2 prox
    l2d = MPIL2(Op=ABDiag_MPI, b=y, x0=x0)

    # L1 prox
    l1 = L1(sigma=1e-1)
    l1d = MPIProxOperator(l1)

    xinv = MPIProximalGradient(l2d, l1d, x0=x0, tau=1e-3, niter=50, show=True)
    assert isinstance(xinv, DistributedArray)
    xinv_array = xinv.asarray()

    As = base_comm.allgather(A)
    if rank == 0:
        ABDiag = BlockDiag([MatrixMult(A, dtype=par["dtype"]) for A in As])
        if par["x0"]:
            x0 = x0_global
        else:
            x0 = cp.zeros(par["nx"] * size, dtype=par["dtype"])
        y1 = ABDiag * x_global

        l2local = L2(Op=ABDiag, b=y1, x0=x0)
        l1local = L1(sigma=1e-1)

        xinv1 = ProximalGradient(
            l2local, l1local, x0=x0, tau=1e-3, niter=50, show=False
        )
        assert_allclose(xinv_array.get(), xinv1.get(), rtol=1e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par2), (par2j), (par3), (par3j), (par4), (par4j)]
)
def test_admml2_scatter(par):
    """ADMML2 with scattered model"""
    cp.random.seed(rank)

    A = cp.random.normal(0, 1, (par["ny"], par["nx"])) + par["imag"] * cp.random.normal(
        0, 1, (par["ny"], par["nx"])
    )
    ABDiag_MPI = MPIBlockDiag(
        ops=[
            pylops.MatrixMult(A, dtype=par["dtype"]),
        ]
    )

    x = DistributedArray(
        global_shape=par["nx"] * size,
        dtype=par["dtype"],
        partition=Partition.SCATTER,
        engine="cupy",
    )
    x[:] = cp.random.normal(1, 10, par["nx"]) + par["imag"] * cp.random.normal(
        10, 10, par["nx"]
    )
    x_global = x.asarray()
    if par["x0"]:
        x0 = DistributedArray(
            global_shape=par["nx"] * size,
            dtype=par["dtype"],
            partition=Partition.SCATTER,
            engine="cupy",
        )
        x0[:] = cp.random.normal(1, 10, par["nx"]) + par["imag"] * cp.random.normal(
            10, 10, par["nx"]
        )
        x0_global = x0.asarray()
    else:
        # Set to 0s if x0 = False
        x0 = DistributedArray(
            global_shape=par["nx"] * size,
            dtype=par["dtype"],
            partition=Partition.SCATTER,
            engine="cupy",
        )
        x0[:] = 0
        x0_global = x0.asarray()

    y = ABDiag_MPI * x

    # Regularizer (just make identity to solve the same problem
    # as ProximalGradient)
    Iopd = MPIBlockDiag(
        ops=[
            pylops.Identity(par["nx"], dtype=par["dtype"]),
        ]
    )

    # L1 prox
    l1 = L1(sigma=1e-1)
    l1d = MPIProxOperator(l1)

    xinv = MPIADMML2(l1d, ABDiag_MPI, y, Iopd, x0=x0, tau=1e-3, niter=50, show=True)[0]
    assert isinstance(xinv, DistributedArray)
    xinv_array = xinv.asarray()

    As = base_comm.allgather(A)
    if rank == 0:
        ABDiag = BlockDiag([MatrixMult(A, dtype=par["dtype"]) for A in As])
        if par["x0"]:
            x0 = x0_global
        else:
            x0 = cp.zeros(par["nx"] * size, dtype=par["dtype"])
        y1 = ABDiag * x_global

        Iop = pylops.Identity(par["nx"] * size, dtype=par["dtype"])
        l1local = L1(sigma=1e-1)

        xinv1 = ADMML2(l1local, ABDiag, y1, Iop, x0=x0, tau=1e-3, niter=50, show=False)[
            0
        ]

        # Pretty high tolerance because a different
        # linear solver is used internally in the
        # serial vs distributed versions of ADMML2
        assert_allclose(xinv_array.get(), xinv1.get(), rtol=1e-3)
