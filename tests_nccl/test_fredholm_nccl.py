"""Test the MPIFredholm1 class
Designed to run with n GPUs (with 1 MPI process per GPU)
$ mpiexec -n 3 pytest test_fredholm_nccl.py --with-mpi

This file employs the same test sets as test_fredholm under NCCL environment
"""
import numpy as np
import cupy as cp
from numpy.testing import assert_allclose
from mpi4py import MPI
import pytest

import pylops
import pylops_mpi

from pylops_mpi import DistributedArray
from pylops_mpi.DistributedArray import local_split, Partition
from pylops_mpi.signalprocessing import MPIFredholm1
from pylops_mpi.utils.dottest import dottest
from pylops_mpi.utils._nccl import initialize_nccl_comm

np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

nccl_comm = initialize_nccl_comm()

par1 = {
    "nsl": 12,
    "ny": 6,
    "nx": 4,
    "nz": 5,
    "usematmul": False,
    "saveGt": True,
    "imag": 0,
    "dtype": "float32",
}  # real, saved Gt
par2 = {
    "nsl": 12,
    "ny": 6,
    "nx": 4,
    "nz": 5,
    "usematmul": True,
    "saveGt": False,
    "imag": 0,
    "dtype": "float32",
}  # real, unsaved Gt
par3 = {
    "nsl": 12,
    "ny": 6,
    "nx": 4,
    "nz": 5,
    "usematmul": False,
    "saveGt": True,
    "imag": 1j,
    "dtype": "complex64",
}  # complex, saved Gt
par4 = {
    "nsl": 12,
    "ny": 6,
    "nx": 4,
    "nz": 5,
    "saveGt": False,
    "usematmul": False,
    "imag": 1j,
    "dtype": "complex64",
}  # complex, unsaved Gt
par5 = {
    "nsl": 12,
    "ny": 6,
    "nx": 4,
    "nz": 1,
    "usematmul": True,
    "saveGt": True,
    "imag": 0,
    "dtype": "float32",
}  # real, saved Gt, nz=1
par6 = {
    "nsl": 12,
    "ny": 6,
    "nx": 4,
    "nz": 1,
    "usematmul": True,
    "saveGt": False,
    "imag": 0,
    "dtype": "float32",
}  # real, unsaved Gt, nz=1


"""Seems to stop next tests from running
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1)])
def test_Gsize1(par):
    #Check error is raised if G has size 1 in any of the ranks
    with pytest.raises(NotImplementedError):
        _ = MPIFredholm1(
                np.ones((1,  par["nx"], par["ny"])),
                nz=par["nz"],
                saveGt=par["saveGt"],
                usematmul=par["usematmul"],
                dtype=par["dtype"],
            )
"""


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_Fredholm1_nccl(par):
    """Fredholm1 operator"""
    np.random.seed(42)

    _F = cp.arange(par["nsl"] * par["nx"] * par["ny"]).reshape(
        par["nsl"], par["nx"], par["ny"]
    ).astype(par["dtype"])
    F = _F - par["imag"] * _F

    # split across ranks
    nsl_rank = local_split((par["nsl"], ), MPI.COMM_WORLD, Partition.SCATTER, 0)
    nsl_ranks = np.concatenate(MPI.COMM_WORLD.allgather(nsl_rank))
    islin_rank = np.insert(np.cumsum(nsl_ranks)[:-1] , 0, 0)[rank]
    islend_rank = np.cumsum(nsl_ranks)[rank]
    Frank = F[islin_rank:islend_rank]

    Fop_MPI = MPIFredholm1(
        Frank,
        nz=par["nz"],
        saveGt=par["saveGt"],
        usematmul=par["usematmul"],
        dtype=par["dtype"],
    )

    x = DistributedArray(global_shape=par["nsl"] * par["ny"] * par["nz"],
                         base_comm_nccl=nccl_comm,
                         partition=pylops_mpi.Partition.BROADCAST,
                         dtype=par["dtype"],
                         engine="cupy")
    x[:] = 1. + par["imag"] * 1.
    x_global = x.asarray()
    # Forward
    y_dist = Fop_MPI @ x
    y = y_dist.asarray()
    # Adjoint
    y_adj_dist = Fop_MPI.H @ y_dist
    y_adj = y_adj_dist.asarray()
    # Dot test
    dottest(Fop_MPI, x, y_dist, par["nsl"] * par["nx"] * par["nz"], par["nsl"] * par["ny"] * par["nz"])

    if rank == 0:
        Fop = pylops.signalprocessing.Fredholm1(
            F.get(),
            nz=par["nz"],
            saveGt=par["saveGt"],
            usematmul=par["usematmul"],
            dtype=par["dtype"],
        )

        assert Fop_MPI.shape == Fop.shape
        y_np = Fop @ x_global.get()
        y_adj_np = Fop.H @ y_np
        assert_allclose(y.get(), y_np, rtol=1e-14)
        assert_allclose(y_adj.get(), y_adj_np, rtol=1e-14)
