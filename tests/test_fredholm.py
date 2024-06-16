import numpy as np
from numpy.testing import assert_allclose
from mpi4py import MPI
import pytest

import pylops
import pylops_mpi

from pylops_mpi import DistributedArray
from pylops_mpi.DistributedArray import local_split, Partition
from pylops_mpi.signalprocessing import MPIFredholm1

np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


par1 = {
    "nsl": 6,
    "ny": 6,
    "nx": 4,
    "nz": 5,
    "usematmul": False,
    "saveGt": True,
    "imag": 0,
    "dtype": "float32",
}  # real, saved Gt
par2 = {
    "nsl": 6,
    "ny": 6,
    "nx": 4,
    "nz": 5,
    "usematmul": True,
    "saveGt": False,
    "imag": 0,
    "dtype": "float32",
}  # real, unsaved Gt
par3 = {
    "nsl": 6,
    "ny": 6,
    "nx": 4,
    "nz": 5,
    "usematmul": False,
    "saveGt": True,
    "imag": 1j,
    "dtype": "complex64",
}  # complex, saved Gt
par4 = {
    "nsl": 6,
    "ny": 6,
    "nx": 4,
    "nz": 5,
    "saveGt": False,
    "usematmul": False,
    "saveGt": False,
    "imag": 1j,
    "dtype": "complex64",
}  # complex, unsaved Gt
par5 = {
    "nsl": 6,
    "ny": 6,
    "nx": 4,
    "nz": 1,
    "usematmul": True,
    "saveGt": True,
    "imag": 0,
    "dtype": "float32",
}  # real, saved Gt, nz=1
par6 = {
    "nsl": 6,
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
def test_Fredholm1(par):
    """Fredholm1 operator"""
    np.random.seed(42)

    _F = np.arange(par["nsl"] * par["nx"] * par["ny"]).reshape(
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
                         partition=pylops_mpi.Partition.BROADCAST,
                         dtype=par["dtype"])
    x[:] = 1. + par["imag"] * 1.
    x_global = x.asarray()
    # Forward
    y_dist = Fop_MPI @ x
    y = y_dist.asarray()
    # Adjoint
    y_adj_dist = Fop_MPI.H @ y_dist
    y_adj = y_adj_dist.asarray()

    if rank == 0:
        Fop = pylops.signalprocessing.Fredholm1(
            F,
            nz=par["nz"],
            saveGt=par["saveGt"],
            usematmul=False,
            dtype=par["dtype"],
        )

        assert Fop_MPI.shape == Fop.shape
        y_np = Fop @ x_global
        y_adj_np = Fop.H @ y_np
        assert_allclose(y, y_np, rtol=1e-14)
        assert_allclose(y_adj, y_adj_np, rtol=1e-14)
