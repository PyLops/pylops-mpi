"""Test proximal solvers
    Designed to run with n processes
    $ mpiexec -n 10 pytest test_proxsolver.py --with-mpi
"""
import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_allclose

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_allclose

    backend = "numpy"
from mpi4py import MPI
import pytest
import pylops
from pylops import (
    BlockDiag,
    MatrixMult,
)
from pyproximal import L1, L2
from pyproximal.optimization.primal import ProximalGradient

from pylops_mpi import DistributedArray, Partition
from pylops_mpi.basicoperators import MPIBlockDiag, MPIVStack
from pylops_mpi.proximal import MPIProxOperator, MPIL2
from pylops_mpi.proximal.optimization.primal import ProximalGradient as MPIProximalGradient
from pylops_mpi.proximal.optimization.primal import ADMML2 as MPIADMML2


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if backend == "cupy":
    device_id = rank % np.cuda.runtime.getDeviceCount()
    np.cuda.Device(device_id).use()

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
    np.random.seed(rank)

    A = np.random.normal(0, 1, (par["ny"], par["nx"])) + par[
        "imag"] * np.random.normal(0, 1, (par["ny"], par["nx"]))
    AVStack_MPI = MPIVStack(ops=[pylops.MatrixMult(A), ])

    x = DistributedArray(global_shape=par['nx'], dtype=par['dtype'],
                         partition=Partition.BROADCAST, engine=backend)
    x[:] = np.random.normal(1, 10, par["nx"]) + \
        par["imag"] * np.random.normal(10, 10, par["nx"])
    x_global = x.asarray()
    if par["x0"]:
        x0 = DistributedArray(global_shape=par['nx'], dtype=par['dtype'],
                              partition=Partition.BROADCAST, engine=backend)
        x0[:] = np.random.normal(1, 10, par["nx"]) + \
            par["imag"] * np.random.normal(10, 10, par["nx"])
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        x0 = DistributedArray(global_shape=par['nx'], dtype=par['dtype'],
                              partition=Partition.BROADCAST, engine=backend)
        x0[:] = 0
        x0_global = x0.asarray()

    y = AVStack_MPI * x

    # L2 prox
    l2d = MPIL2(
        Op=AVStack_MPI, b=y, x0=x0)

    # L1 prox
    l1 = L1(sigma=1e-1)
    l1d = MPIProxOperator(l1)

    xinv = MPIProximalGradient(
        l2d, l1d, x0=x0, tau=1e-3, niter=400, show=True)
    assert isinstance(xinv, DistributedArray)
    xinv_array = xinv.asarray()
    
    As = np.vstack(comm.allgather(A))
    if rank == 0:
        AVStack = MatrixMult(As)
        if par["x0"]:
            x0 = x0_global
        else:
            x0 = np.zeros(par['nx'], dtype=par['dtype'])
        y1 = AVStack * x_global

        l2local = L2(Op=AVStack, b=y1, x0=x0)
        l1local = L1(sigma=1e-1)

        xinv1 = ProximalGradient(
            l2local, l1local, x0=x0, tau=1e-3,
            niter=400, show=False)
        assert_allclose(xinv_array, xinv1, rtol=1e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par2), (par2j), (par3), (par3j), (par4), (par4j)]
)
def test_proximalgradient_scatter(par):
    """ProximalGradient with broabcasted model"""
    np.random.seed(rank)

    A = np.random.normal(0, 1, (par["ny"], par["nx"])) + par[
        "imag"] * np.random.normal(0, 1, (par["ny"], par["nx"]))
    ABDiag_MPI = MPIBlockDiag(ops=[pylops.MatrixMult(A), ])

    x = DistributedArray(global_shape=par['nx'] * size, dtype=par['dtype'],
                         partition=Partition.SCATTER, engine=backend)
    x[:] = np.random.normal(1, 10, par["nx"]) + \
        par["imag"] * np.random.normal(10, 10, par["nx"])
    x_global = x.asarray()
    if par["x0"]:
        x0 = DistributedArray(global_shape=par['nx'] * size, dtype=par['dtype'],
                              partition=Partition.SCATTER, engine=backend)
        x0[:] = np.random.normal(1, 10, par["nx"]) + \
            par["imag"] * np.random.normal(10, 10, par["nx"])
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        x0 = DistributedArray(global_shape=par['nx'] * size, dtype=par['dtype'],
                              partition=Partition.SCATTER, engine=backend)
        x0[:] = 0
        x0_global = x0.asarray()

    y = ABDiag_MPI * x

    # L2 prox
    l2d = MPIL2(
        Op=ABDiag_MPI, b=y, x0=x0)

    # L1 prox
    l1 = L1(sigma=1e-1)
    l1d = MPIProxOperator(l1)

    xinv = MPIProximalGradient(
        l2d, l1d, x0=x0, tau=1e-3, niter=400, show=True)
    assert isinstance(xinv, DistributedArray)
    xinv_array = xinv.asarray()

    As = comm.allgather(A)
    if rank == 0:
        ABDiag = BlockDiag([MatrixMult(A) for A in As])
        if par["x0"]:
            x0 = x0_global
        else:
            x0 = np.zeros(par['nx'] * size, dtype=par['dtype'])
        y1 = ABDiag * x_global

        l2local = L2(Op=ABDiag, b=y1, x0=x0)
        l1local = L1(sigma=1e-1)

        xinv1 = ProximalGradient(
            l2local, l1local, x0=x0, tau=1e-3,
            niter=400, show=False)
        assert_allclose(xinv_array, xinv1, rtol=1e-12)
