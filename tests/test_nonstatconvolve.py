"""Test the MPINonStationaryConvolve1D class
    Designed to run with n processes
    $ mpiexec -n 10 pytest test_nonstatconvolve.py --with-mpi
"""
import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupyx.scipy.signal.windows import triang
    backend = "cupy"
else:
    import numpy as np
    from scipy.signal.windows import triang
    backend = "numpy"
from mpi4py import MPI
import pytest
import pylops
from numpy.testing import assert_allclose

import pylops_mpi
from pylops_mpi import DistributedArray
from pylops_mpi.signalprocessing import MPINonStationaryConvolve1D
from pylops_mpi.utils.dottest import dottest

np.random.seed(42)
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
if backend == "cupy":
    device_id = rank % np.cuda.runtime.getDeviceCount()
    np.cuda.Device(device_id).use()

# filters
nfilts = (5, 7)
nfilts3 = (5, 5, 7)
filts_local = 2

h1 = triang(nfilts[0], sym=True)
h1s = np.tile(h1[None], (size * filts_local, 1))

par1_1d = {
    "nz": 32,
    "nx": 64,
    "axis": 0,
    "dtype": np.float64,
}  # first direction
par2_1d = {
    "nz": 32,
    "nx": 64,
    "axis": 1,
    "dtype": np.float64,
}  # second direction


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1_1d)])
def test_even_filter(par):
    """Check error is raised if filter has even size"""
    with pytest.raises(ValueError, match="filters hs must have odd length"):
        n_between_h = par["nz"] // filts_local
        ih = par["nz"] // (2 * filts_local) + \
            np.arange(0, par["nz"] * size, n_between_h)

        _ = MPINonStationaryConvolve1D(
            dims=par["nx"],
            hs=h1s[..., :-1],
            ih=ih,
        )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1_1d)])
def test_ih_irregular(par):
    """Check error is raised if ih (or ihx/ihz) are irregularly sampled"""
    with pytest.raises(ValueError, match="must be regularly sampled"):
        n_between_h = par["nz"] // filts_local
        ih = par["nz"] // (2 * filts_local) + \
            np.arange(0, par["nz"] * size, n_between_h)
        ih[0] = ih[0] + 1
        _ = MPINonStationaryConvolve1D(
            dims=par["nx"],
            hs=h1s,
            ih=ih,
        )


@pytest.mark.parametrize("par", [(par1_1d), (par2_1d)])
def test_NonStationaryConvolve1D(par):
    """Dot-test and inversion for NonStationaryConvolve1D operator"""
    # 1D
    if par["axis"] == 0:
        n_between_h = par["nz"] // filts_local
        ih = par["nz"] // (2 * filts_local) + \
            np.arange(0, par["nz"] * size, n_between_h)
        Cop_MPI = MPINonStationaryConvolve1D(
            dims=par["nz"] * size,
            hs=h1s,
            ih=ih,
            dtype=par["dtype"]
        )
        
        x_global = np.zeros(size * par["nz"], dtype=np.float64)
        x_global[ih] = 1.0

        x = DistributedArray(global_shape=size * par["nz"],
                             partition=pylops_mpi.Partition.SCATTER,
                             dtype=par["dtype"], engine=backend)
        x.local_array[:] = x_global[par["nz"] * rank: par["nz"] * (rank + 1)]
        # Forward
        y_dist = Cop_MPI @ x
        y = y_dist.asarray()
        # Adjoint
        y_adj_dist = Cop_MPI.H @ y_dist
        y_adj = y_adj_dist.asarray()
        # Dot test
        dottest(Cop_MPI, x, y_dist, size * par["nz"], size * par["nz"])

        if rank == 0:
            Cop = pylops.signalprocessing.NonStationaryConvolve1D(
                dims=par["nz"] * size, hs=h1s, ih=ih,
                dtype=par['dtype'])
            assert Cop_MPI.shape == Cop.shape
            y_np = Cop @ x_global
            y_adj_np = Cop.H @ y_np
            assert_allclose(y, y_np, rtol=1e-14)
            assert_allclose(y_adj, y_adj_np, rtol=1e-14)
