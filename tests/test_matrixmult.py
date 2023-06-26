import numpy as np
from numpy.testing import assert_allclose

np.random.seed(42)
import pytest

import pylops
from pylops_mpi import MPIMatrixMult, DistributedArray

par1 = {"ny": 11, "nx": 11, "dtype": np.float64}
par2 = {"ny": 21, "nx": 11, "dtype": np.float64}
par1j = {"ny": 11, "nx": 11, "dtype": np.float64}
par2j = {"ny": 21, "nx": 11, "dtype": np.float64}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_matmult(par):
    A = np.random.normal(1, 10, (par['nx'], par['ny']))
    Mop = pylops.MatrixMult(A=A)

    # kind="all"
    arr = DistributedArray.to_dist(x=A)
    Mop_all = MPIMatrixMult(A=arr.local_array, kind="all")

    # kind="master"
    Mop_master = MPIMatrixMult(A=A, kind="master")

    x = np.arange(par['ny'])
    y = np.arange(par['nx'])
    if Mop_master.rank == 0:
        assert_allclose(Mop_all * x, Mop * x, rtol=1e-12)
        assert_allclose(Mop_master * x, Mop * x, rtol=1e-12)
        assert_allclose(Mop_all.H * y, Mop.H * y, rtol=1e-12)
        assert_allclose(Mop_master.H * y, Mop.H * y, rtol=1e-12)
    else:
        assert_allclose(Mop_all * x, Mop * x, rtol=1e-12)
        assert_allclose(Mop_all.H * y, Mop.H * y, rtol=1e-12)
