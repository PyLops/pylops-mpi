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
    MPIVStack,
    Partition
)

np.random.seed(42)
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

par1 = {
    "ny": 11,
    "nx": 11,
    "imag": 0,
    "x0": False,
    "dtype": "float128",
}  # square real, zero initial guess
par2 = {
    "ny": 11,
    "nx": 11,
    "imag": 0,
    "x0": True,
    "dtype": "float128",
}  # square real, non-zero initial guess
par3 = {
    "ny": 31,
    "nx": 11,
    "imag": 0,
    "x0": False,
    "dtype": "float128",
}  # overdetermined real, zero initial guess
par4 = {
    "ny": 31,
    "nx": 11,
    "imag": 0,
    "x0": True,
    "dtype": "float128",
}  # overdetermined real, non-zero initial guess
par1j = {
    "ny": 11,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex256",
}  # square complex, zero initial guess
par2j = {
    "ny": 11,
    "nx": 11,
    "imag": 1j,
    "x0": True,
    "dtype": "complex256",
}  # square complex, non-zero initial guess
par3j = {
    "ny": 31,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex256",
}  # overdetermined complex, zero initial guess
par4j = {
    "ny": 31,
    "nx": 11,
    "imag": 1j,
    "x0": True,
    "dtype": "complex256",
}  # overdetermined complex, non-zero initial guess


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par2), (par2j), (par3), (par3j), (par4), (par4j)]
)
def test_cg(par):
    """CG with MPIBlockDiag"""
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
        x0 = None
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
        assert_allclose(xinv_array, xinv1, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par2), (par2j), (par3), (par3j), (par4), (par4j)]
)
def test_cgls(par):
    """CGLS with MPIBlockDiag"""
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
        x0 = None
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
        assert_allclose(xinv_array, xinv1, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par2), (par2j), (par3), (par3j), (par4), (par4j)]
)
def test_cgls_broadcastdata(par):
    """CGLS with broadcasted data vector"""
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
        x0 = None

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
        assert_allclose(xinv_array, xinv1, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par2), (par2j), (par3), (par3j), (par4), (par4j)]
)
def test_cgls_broadcastmodel(par):
    """CGLS with broadcasted model vector"""
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
        x0 = None

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
        assert_allclose(xinv_array, xinv1, rtol=1e-14)
