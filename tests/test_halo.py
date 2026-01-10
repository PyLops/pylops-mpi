import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    backend = "cupy"
else:
    import numpy as np
    backend = "numpy"
import numpy as npp
from mpi4py import MPI
import pytest

import pylops_mpi
from pylops_mpi.basicoperators.Halo import MPIHalo
from pylops_mpi.utils.dottest import dottest

np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()
if backend == "cupy":
    device_id = rank % np.cuda.runtime.getDeviceCount()
    np.cuda.Device(device_id).use()


@pytest.mark.mpi(min_size=8)
def test_halo_dottest_plot_config():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    p_prime = int(round(size ** (1 / 3)))
    if p_prime ** 3 != size:
        pytest.skip("MPI size must be a perfect cube for 3D halo grid")

    gdim = (4 * p_prime, 4 * p_prime, 4 * p_prime)
    g_shape = (p_prime, p_prime, p_prime)
    halo = 1

    halo_op = MPIHalo(
        dims=gdim,
        halo=halo,
        proc_grid_shape=g_shape,
        comm=comm,
        dtype=np.float64,
    )

    x_dist = pylops_mpi.DistributedArray(
        global_shape=npp.prod(gdim),
        base_comm=comm,
        partition=pylops_mpi.Partition.SCATTER,
        engine=backend,
        dtype=np.float64,
    )
    x_dist[:] = np.random.normal(0.0, 1.0, x_dist.local_array.shape)

    y_dist = pylops_mpi.DistributedArray(
        global_shape=halo_op.shape[0],
        base_comm=comm,
        partition=pylops_mpi.Partition.SCATTER,
        engine=backend,
        dtype=np.float64,
    )
    y_dist[:] = np.random.normal(0.0, 1.0, y_dist.local_array.shape)

    dottest(halo_op, x_dist, y_dist)
