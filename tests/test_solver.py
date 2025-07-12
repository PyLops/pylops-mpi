"""Test solvers
    Designed to run with n processes
    $ mpiexec -n 10 pytest test_solver.py --with-mpi
"""
import numpy as np
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
def test_cg(par):
    """CG with MPIBlockDiag"""
    np.random.seed(42)
    
    A = np.ones((par["ny"], par["nx"])) + par[
        "imag"] * np.ones((par["ny"], par["nx"]))
    Aop = MatrixMult(np.conj(A.T) @ A, dtype=par['dtype'])
    # To make MPIBlockDiag a positive definite matrix
    BDiag_MPI = MPIBlockDiag(ops=[Aop, ])

    x = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    x[:] = np.random.normal(1, 10, par["nx"]) + par["imag"] * np.random.normal(10, 10, par["nx"])
    x_global = x.asarray()
    if par["x0"]:
        x0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
        x0[:] = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            10, 10, par["nx"]
        )
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        x0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
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
            x0 = x0_global
        else:
            x0 = None
        y1 = BDiag * x_global
        xinv1 = pylops.cg(BDiag, y1, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
        assert_allclose(xinv_array, xinv1, rtol=1e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par2), (par2j), (par3), (par3j), (par4), (par4j)]
)
def test_cgls(par):
    """CGLS with MPIBlockDiag"""
    np.random.seed(42)

    A = np.ones((par["ny"], par["nx"])) + par[
        "imag"] * np.ones((par["ny"], par["nx"]))
    Aop = MatrixMult(np.conj(A.T) @ A + 1e-5 * np.eye(par["nx"], dtype=par['dtype']),
                     dtype=par['dtype'])
    # To make MPIBlockDiag a positive definite matrix
    BDiag_MPI = MPIBlockDiag(ops=[Aop, ])

    x = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    x[:] = np.random.normal(1, 10, par["nx"]) + par["imag"] * np.random.normal(10, 10, par["nx"])
    x_global = x.asarray()
    if par["x0"]:
        x0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
        x0[:] = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            10, 10, par["nx"]
        )
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        x0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
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
            x0 = x0_global
        else:
            x0 = None
        y1 = BDiag * x_global
        xinv1 = pylops.cgls(BDiag, y1, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
        assert_allclose(xinv_array, xinv1, rtol=1e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par2), (par2j), (par3), (par3j), (par4), (par4j)]
)
def test_cgls_broadcastdata(par):
    """CGLS with broadcasted data vector"""
    np.random.seed(42)

    A = (rank + 1) * np.ones((par["ny"], par["nx"])) + (rank + 2) * par[
        "imag"
    ] * np.ones((par["ny"], par["nx"]))
    Aop = MatrixMult(A, dtype=par["dtype"])
    HStack_MPI = MPIHStack(ops=[Aop, ])

    x = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    x[:] = np.random.normal(1, 10, par['nx']) + par["imag"] * np.random.normal(10, 10, par['nx'])
    x_global = x.asarray()
    if par["x0"]:
        x0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
        x0[:] = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            10, 10, par["nx"]
        )
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        x0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
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
            x0 = x0_global
        else:
            x0 = None
        y1 = Hstack @ x_global
        xinv1 = pylops.cgls(Hstack, y1, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
        assert_allclose(xinv_array, xinv1, rtol=1e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par2), (par2j), (par3), (par3j), (par4), (par4j)]
)
def test_cgls_broadcastmodel(par):
    """CGLS with broadcasted model vector"""
    np.random.seed(42)

    A = np.ones((par["ny"], par["nx"])) + par[
        "imag"] * np.ones((par["ny"], par["nx"]))
    Aop = MatrixMult(np.conj(A.T) @ A + 1e-5 * np.eye(par["nx"], dtype=par['dtype']),
                     dtype=par['dtype'])
    # To make MPIVStack a positive definite matrix
    VStack_MPI = MPIVStack(ops=[Aop, ])

    x = DistributedArray(global_shape=par['nx'], dtype=par['dtype'], partition=Partition.BROADCAST)
    x[:] = np.random.normal(1, 10, par['nx']) + par["imag"] * np.random.normal(10, 10, par['nx'])
    x_global = x.asarray()
    if par["x0"]:
        x0 = DistributedArray(global_shape=par['nx'], dtype=par['dtype'], partition=Partition.BROADCAST)
        x0[:] = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            10, 10, par["nx"]
        )
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        x0 = DistributedArray(global_shape=par['nx'], dtype=par['dtype'], partition=Partition.BROADCAST)
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
            x0 = x0_global
        else:
            x0 = None
        y1 = Vstack @ x_global
        xinv1 = pylops.cgls(Vstack, y1, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
        assert_allclose(xinv_array, xinv1, rtol=1e-13)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par2), (par2j), (par3), (par3j), (par4), (par4j)]
)
def test_cg_stacked(par):
    """CG with MPIStackedBlockDiag"""
    np.random.seed(42)

    A = np.ones((par["ny"], par["nx"])) + par[
        "imag"] * np.ones((par["ny"], par["nx"]))
    Aop = MatrixMult(np.conj(A.T) @ A + 1e-5 * np.eye(par["nx"], dtype=par['dtype']),
                     dtype=par['dtype'])
    # To make positive definite matrix
    BDiag_MPI = MPIBlockDiag(ops=[Aop, ])
    StackedBDiag_MPI = MPIStackedBlockDiag(ops=[BDiag_MPI, BDiag_MPI])

    dist1 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    dist1[:] = np.random.normal(1, 10, par["nx"]) + par["imag"] * np.random.normal(10, 10, par["nx"])
    dist2 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    dist2[:] = np.random.normal(5, 10, par["nx"]) + par["imag"] * np.random.normal(50, 10, par["nx"])
    x = StackedDistributedArray([dist1, dist2])
    x_global = x.asarray()

    if par["x0"]:
        dist1_0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
        dist1_0[:] = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            10, 10, par["nx"]
        )
        dist2_0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
        dist2_0[:] = np.random.normal(10, 10, par["nx"]) + par["imag"] * np.random.normal(
            0, 10, par["nx"]
        )
        x0 = StackedDistributedArray([dist1_0, dist2_0])
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        dist1_0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
        dist1_0[:] = 0
        dist2_0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
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
            x0 = x0_global
        else:
            x0 = None
        y1 = StackedBDiag * x_global
        xinv1 = pylops.cg(StackedBDiag, y1, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
        assert_allclose(xinv_array, xinv1, rtol=1e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par2), (par2j), (par3), (par3j), (par4), (par4j)]
)
def test_cgls_stacked(par):
    """CGLS with MPIStackedBlockDiag"""
    np.random.seed(42)

    A = np.ones((par["ny"], par["nx"])) + par[
        "imag"] * np.ones((par["ny"], par["nx"]))
    Aop = MatrixMult(np.conj(A.T) @ A + 1e-5 * np.eye(par["nx"], dtype=par['dtype']),
                     dtype=par['dtype'])
    # To make positive definite matrix
    BDiag_MPI = MPIBlockDiag(ops=[Aop, ])
    VStack_MPI = MPIVStack(ops=[Aop, ])
    StackedBDiag_MPI = MPIStackedBlockDiag(ops=[BDiag_MPI, VStack_MPI])

    dist1 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    dist1[:] = np.random.normal(1, 10, par["nx"]) + par["imag"] * np.random.normal(10, 10, par["nx"])
    dist2 = DistributedArray(global_shape=par['nx'], partition=Partition.BROADCAST, dtype=par['dtype'])
    dist2[:] = np.random.normal(5, 10, dist2.local_shape) + par["imag"] * np.random.normal(50, 10, dist2.local_shape)
    x = StackedDistributedArray([dist1, dist2])
    x_global = x.asarray()

    if par["x0"]:
        dist1_0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
        dist1_0[:] = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            10, 10, par["nx"]
        )
        dist2_0 = DistributedArray(global_shape=par['nx'], partition=Partition.BROADCAST, dtype=par['dtype'])
        dist2_0[:] = np.random.normal(10, 10, dist2_0.local_shape) + par["imag"] * np.random.normal(
            0, 10, dist2_0.local_shape
        )
        x0 = StackedDistributedArray([dist1_0, dist2_0])
        x0_global = x0.asarray()
    else:
        # Set TO 0s if x0 = False
        dist1_0 = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
        dist1_0[:] = 0
        dist2_0 = DistributedArray(global_shape=par['nx'], partition=Partition.BROADCAST, dtype=par['dtype'])
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
            x0 = x0_global
        else:
            x0 = None
        y1 = StackedBDiag * x_global
        xinv1 = pylops.cgls(StackedBDiag, y1, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
        assert_allclose(xinv_array, xinv1, rtol=1e-12)
