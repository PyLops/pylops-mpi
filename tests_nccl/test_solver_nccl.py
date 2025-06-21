"""Test solvers
    Designed to run with n GPUs (with 1 MPI process per GPU)
    $ mpiexec -n 10 pytest test_solver_nccl.py --with-mpi

This file employs the same test sets as test_solver_nccl under NCCL environment
"""
import numpy as np
import cupy as cp
from numpy.testing import assert_allclose
from mpi4py import MPI
import pytest
import pylops
from pylops import (
    MatrixMult,
    BlockDiag,
    HStack,
    VStack
)

from pylops_mpi import (
    cg,
    cgls,
    DistributedArray,
    MPIBlockDiag,
    MPIHStack,
    MPIStackedBlockDiag,
    MPIVStack,
    Partition,
    StackedDistributedArray
)

from pylops_mpi.utils._nccl import initialize_nccl_comm

nccl_comm = initialize_nccl_comm()

np.random.seed(42)
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

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

# par1j = {
#     "ny": 11,
#     "nx": 11,
#     "imag": 1j,
#     "x0": False,
#     "dtype": "complex128",
# }  # square complex, zero initial guess
# par2j = {
#     "ny": 11,
#     "nx": 11,
#     "imag": 1j,
#     "x0": True,
#     "dtype": "complex128",
# }  # square complex, non-zero initial guess
# par3j = {
#     "ny": 31,
#     "nx": 11,
#     "imag": 1j,
#     "x0": False,
#     "dtype": "complex128",
# }  # overdetermined complex, zero initial guess
# par4j = {
#     "ny": 31,
#     "nx": 11,
#     "imag": 1j,
#     "x0": True,
#     "dtype": "complex128",
# }  # overdetermined complex, non-zero initial guess


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4)]
)
def test_cg_nccl(par):
    """CG with MPIBlockDiag"""
    A = cp.ones((par["ny"], par["nx"])) + par[
        "imag"] * cp.ones((par["ny"], par["nx"]))
    Aop = MatrixMult(cp.conj(A.T) @ A, dtype=par['dtype'])
    # To make MPIBlockDiag a positive definite matrix
    BDiag_MPI = MPIBlockDiag(ops=[Aop, ])

    x = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    x[:] = cp.random.normal(1, 10, par["nx"]) + par["imag"] * cp.random.normal(10, 10, par["nx"])
    x_global = x.asarray()
    if par["x0"]:
        x0 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
        x0[:] = cp.random.normal(0, 10, par["nx"]) + par["imag"] * cp.random.normal(
            10, 10, par["nx"]
        )
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        x0 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
        x0[:] = 0
        x0_global = x0.asarray()
    y = BDiag_MPI * x
    xinv = cg(BDiag_MPI, y, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
    assert isinstance(xinv, DistributedArray)
    xinv_array = xinv.asarray()
    if rank == 0:
        mats = [np.ones(shape=(par["ny"], par["nx"])) + par[
            "imag"] * np.ones(shape=(par["ny"], par["nx"])) for i in range(size)]
        ops = [MatrixMult(np.conj(mats[i].T) @ mats[i], dtype=par['dtype']) for i in range(size)]
        # To make BlockDiag a positive definite matrix
        BDiag = BlockDiag(ops=ops, forceflat=True)
        if par["x0"]:
            x0 = x0_global.get()
        else:
            x0 = None
        y1 = BDiag * x_global.get()
        xinv1 = pylops.cg(BDiag, y1, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
        assert_allclose(xinv_array.get(), xinv1, rtol=1e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4)]
)
def test_cgls_nccl(par):
    """CGLS with MPIBlockDiag"""
    A = cp.ones((par["ny"], par["nx"])) + par[
        "imag"] * cp.ones((par["ny"], par["nx"]))
    Aop = MatrixMult(cp.conj(A.T) @ A + 1e-5 * cp.eye(par["nx"], dtype=par['dtype']),
                     dtype=par['dtype'])
    # To make MPIBlockDiag a positive definite matrix
    BDiag_MPI = MPIBlockDiag(ops=[Aop, ])

    x = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    x[:] = cp.random.normal(1, 10, par["nx"]) + par["imag"] * cp.random.normal(10, 10, par["nx"])
    x_global = x.asarray()
    if par["x0"]:
        x0 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
        x0[:] = cp.random.normal(0, 10, par["nx"]) + par["imag"] * cp.random.normal(
            10, 10, par["nx"]
        )
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        x0 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
        x0[:] = 0
        x0_global = x0.asarray()
    y = BDiag_MPI * x
    xinv = cgls(BDiag_MPI, y, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
    assert isinstance(xinv, DistributedArray)
    xinv_array = xinv.asarray()
    if rank == 0:
        mats = [np.ones(shape=(par["ny"], par["nx"])) + par[
            "imag"] * np.ones(shape=(par["ny"], par["nx"])) for i in range(size)]
        ops = [MatrixMult(np.conj(mats[i].T) @ mats[i] + 1e-5 * np.eye(par["nx"], dtype=par['dtype']),
                          dtype=par['dtype']) for i in range(size)]
        # To make BlockDiag a positive definite matrix
        BDiag = BlockDiag(ops=ops, forceflat=True)
        if par["x0"]:
            x0 = x0_global.get()
        else:
            x0 = None
        y1 = BDiag * x_global.get()
        xinv1 = pylops.cgls(BDiag, y1, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
        assert_allclose(xinv_array.get(), xinv1, rtol=1e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4)]
)
def test_cgls_broadcastdata_nccl(par):
    """CGLS with broadcasted data vector"""
    A = (rank + 1) * cp.ones((par["ny"], par["nx"])) + (rank + 2) * par[
        "imag"
    ] * cp.ones((par["ny"], par["nx"]))
    Aop = MatrixMult(A, dtype=par["dtype"])
    HStack_MPI = MPIHStack(ops=[Aop, ])

    x = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    x[:] = cp.random.normal(1, 10, par['nx']) + par["imag"] * cp.random.normal(10, 10, par['nx'])
    x_global = x.asarray()
    if par["x0"]:
        x0 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
        x0[:] = cp.random.normal(0, 10, par["nx"]) + par["imag"] * cp.random.normal(
            10, 10, par["nx"]
        )
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        x0 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
        x0[:] = 0
        x0_global = x0.asarray()
    y = HStack_MPI @ x
    assert y.partition is Partition.BROADCAST

    xinv = cgls(HStack_MPI, y, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
    assert isinstance(xinv, DistributedArray)
    xinv_array = xinv.asarray()
    if rank == 0:
        ops = [MatrixMult((i + 1) * np.ones((par["ny"], par["nx"])) + (i + 2) * par[
            "imag"
        ] * np.ones((par["ny"], par["nx"])), dtype=par['dtype']) for i in range(size)]
        Hstack = HStack(ops=ops, forceflat=True)
        if par["x0"]:
            x0 = x0_global.get()
        else:
            x0 = None
        y1 = Hstack @ x_global.get()
        xinv1 = pylops.cgls(Hstack, y1, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
        assert_allclose(xinv_array.get(), xinv1, rtol=1e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4)]
)
def test_cgls_broadcastmodel_nccl(par):
    """CGLS with broadcasted model vector"""
    A = cp.ones((par["ny"], par["nx"])) + par[
        "imag"] * cp.ones((par["ny"], par["nx"]))
    Aop = MatrixMult(cp.conj(A.T) @ A + 1e-5 * cp.eye(par["nx"], dtype=par['dtype']),
                     dtype=par['dtype'])
    # To make MPIVStack a positive definite matrix
    VStack_MPI = MPIVStack(ops=[Aop, ])

    x = DistributedArray(global_shape=par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], partition=Partition.BROADCAST, engine="cupy")
    x[:] = cp.random.normal(1, 10, par['nx']) + par["imag"] * cp.random.normal(10, 10, par['nx'])
    x_global = x.asarray()
    if par["x0"]:
        x0 = DistributedArray(global_shape=par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], partition=Partition.BROADCAST, engine="cupy")
        x0[:] = cp.random.normal(0, 10, par["nx"]) + par["imag"] * cp.random.normal(
            10, 10, par["nx"]
        )
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        x0 = DistributedArray(global_shape=par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], partition=Partition.BROADCAST, engine="cupy")
        x0[:] = 0
        x0_global = x0.asarray()
    y = VStack_MPI @ x
    assert y.partition is Partition.SCATTER

    xinv = cgls(VStack_MPI, y, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
    assert isinstance(xinv, DistributedArray)
    xinv_array = xinv.asarray()
    if rank == 0:
        mats = [np.ones(shape=(par["ny"], par["nx"])) + par[
            "imag"] * np.ones(shape=(par["ny"], par["nx"])) for i in range(size)]
        ops = [MatrixMult(np.conj(mats[i].T) @ mats[i] + 1e-5 * np.eye(par["nx"], dtype=par['dtype']),
                          dtype=par['dtype']) for i in range(size)]
        # To make VStack a positive definite matrix
        Vstack = VStack(ops=ops, forceflat=True)
        if par["x0"]:
            x0 = x0_global.get()
        else:
            x0 = None
        y1 = Vstack @ x_global.get()
        xinv1 = pylops.cgls(Vstack, y1, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
        assert_allclose(xinv_array.get(), xinv1, rtol=1e-13)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4)]
)
def test_cg_stacked_nccl(par):
    """CG with MPIStackedBlockDiag"""
    A = cp.ones((par["ny"], par["nx"])) + par[
        "imag"] * cp.ones((par["ny"], par["nx"]))
    Aop = MatrixMult(cp.conj(A.T) @ A + 1e-5 * cp.eye(par["nx"], dtype=par['dtype']),
                     dtype=par['dtype'])
    # To make positive definite matrix
    BDiag_MPI = MPIBlockDiag(ops=[Aop, ])
    StackedBDiag_MPI = MPIStackedBlockDiag(ops=[BDiag_MPI, BDiag_MPI])

    dist1 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    dist1[:] = cp.random.normal(1, 10, par["nx"]) + par["imag"] * cp.random.normal(10, 10, par["nx"])
    dist2 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    dist2[:] = cp.random.normal(5, 10, par["nx"]) + par["imag"] * cp.random.normal(50, 10, par["nx"])
    x = StackedDistributedArray([dist1, dist2])
    x_global = x.asarray()

    if par["x0"]:
        dist1_0 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
        dist1_0[:] = cp.random.normal(0, 10, par["nx"]) + par["imag"] * cp.random.normal(
            10, 10, par["nx"]
        )
        dist2_0 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
        dist2_0[:] = cp.random.normal(10, 10, par["nx"]) + par["imag"] * cp.random.normal(
            0, 10, par["nx"]
        )
        x0 = StackedDistributedArray([dist1_0, dist2_0])
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        dist1_0 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
        dist1_0[:] = 0
        dist2_0 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
        dist2_0[:] = 0
        x0 = StackedDistributedArray([dist1_0, dist2_0])
        x0_global = x0.asarray()

    y = StackedBDiag_MPI * x
    xinv = cg(StackedBDiag_MPI, y, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
    assert isinstance(xinv, StackedDistributedArray)
    xinv_array = xinv.asarray()

    if rank == 0:
        mats = [np.ones(shape=(par["ny"], par["nx"])) + par[
            "imag"] * np.ones(shape=(par["ny"], par["nx"])) for i in range(size)]
        ops = [MatrixMult(np.conj(mats[i].T) @ mats[i] + 1e-5 * np.eye(par["nx"], dtype=par['dtype']),
                          dtype=par['dtype']) for i in range(size)]
        # To make positive definite matrix
        BDiag = BlockDiag(ops=ops, forceflat=True)
        StackedBDiag = BlockDiag(ops=[BDiag, BDiag], forceflat=True)
        if par["x0"]:
            x0 = x0_global.get()
        else:
            x0 = None
        y1 = StackedBDiag * x_global.get()
        xinv1 = pylops.cg(StackedBDiag, y1, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
        assert_allclose(xinv_array.get(), xinv1, rtol=1e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4)]
)
def test_cgls_stacked_nccl(par):
    """CGLS with MPIStackedBlockDiag"""
    A = cp.ones((par["ny"], par["nx"])) + par[
        "imag"] * cp.ones((par["ny"], par["nx"]))
    Aop = MatrixMult(cp.conj(A.T) @ A + 1e-5 * cp.eye(par["nx"], dtype=par['dtype']),
                     dtype=par['dtype'])
    # To make positive definite matrix
    BDiag_MPI = MPIBlockDiag(ops=[Aop, ])
    VStack_MPI = MPIVStack(ops=[Aop, ])
    StackedBDiag_MPI = MPIStackedBlockDiag(ops=[BDiag_MPI, VStack_MPI])

    dist1 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    dist1[:] = cp.random.normal(1, 10, par["nx"]) + par["imag"] * cp.random.normal(10, 10, par["nx"])
    dist2 = DistributedArray(global_shape=par['nx'], base_comm_nccl=nccl_comm, partition=Partition.BROADCAST, dtype=par['dtype'], engine="cupy")
    dist2[:] = cp.random.normal(5, 10, dist2.local_shape) + par["imag"] * cp.random.normal(50, 10, dist2.local_shape)
    x = StackedDistributedArray([dist1, dist2])
    x_global = x.asarray()

    if par["x0"]:
        dist1_0 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
        dist1_0[:] = cp.random.normal(0, 10, par["nx"]) + par["imag"] * cp.random.normal(
            10, 10, par["nx"]
        )
        dist2_0 = DistributedArray(global_shape=par['nx'], base_comm_nccl=nccl_comm, partition=Partition.BROADCAST, dtype=par['dtype'], engine="cupy")
        dist2_0[:] = cp.random.normal(10, 10, dist2_0.local_shape) + par["imag"] * cp.random.normal(
            0, 10, dist2_0.local_shape
        )
        x0 = StackedDistributedArray([dist1_0, dist2_0])
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        dist1_0 = DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
        dist1_0[:] = 0
        dist2_0 = DistributedArray(global_shape=par['nx'], base_comm_nccl=nccl_comm, partition=Partition.BROADCAST, dtype=par['dtype'], engine="cupy")
        dist2_0[:] = 0
        x0 = StackedDistributedArray([dist1_0, dist2_0])
        x0_global = x0.asarray()

    y = StackedBDiag_MPI * x
    xinv = cgls(StackedBDiag_MPI, y, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
    assert isinstance(xinv, StackedDistributedArray)
    xinv_array = xinv.asarray()

    if rank == 0:
        mats = [np.ones(shape=(par["ny"], par["nx"])) + par[
            "imag"] * np.ones(shape=(par["ny"], par["nx"])) for i in range(size)]
        ops = [MatrixMult(np.conj(mats[i].T) @ mats[i] + 1e-5 * np.eye(par["nx"], dtype=par['dtype']),
                          dtype=par['dtype']) for i in range(size)]
        # To make positive definite matrix
        BDiag = BlockDiag(ops=ops, forceflat=True)
        V_Stack = VStack(ops=ops, forceflat=True)
        StackedBDiag = BlockDiag(ops=[BDiag, V_Stack], forceflat=True)
        if par["x0"]:
            x0 = x0_global.get()
        else:
            x0 = None
        y1 = StackedBDiag * x_global.get()
        xinv1 = pylops.cgls(StackedBDiag, y1, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
        assert_allclose(xinv_array.get(), xinv1, rtol=1e-12)
