import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    backend = "cupy"
else:
    import numpy as np
    backend = "numpy"
from mpi4py import MPI
import pytest
import pylops
from numpy.testing import assert_allclose

import pylops_mpi
from pylops_mpi.basicoperators.Halo import MPIHalo
from pylops_mpi.utils.dottest import dottest

np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()
if backend == "cupy":
    device_id = rank % np.cuda.runtime.getDeviceCount()
    np.cuda.Device(device_id).use()


@pytest.mark.mpi(min_size=2)
def test_halo():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    nlocal = 16
    n = nlocal * size
    halo = 1

    halo_op = MPIHalo(
        dims=(n,),
        halo=halo,
        proc_grid_shape=(size,),
        comm=comm,
        dtype=np.float64,
    )

    x_dist = pylops_mpi.DistributedArray(
        global_shape=n,
        base_comm=comm,
        partition=pylops_mpi.Partition.SCATTER,
        engine=backend,
        dtype=np.float64,
    )
    x_dist[:] = np.random.normal(0.0, 1.0, x_dist.local_array.shape)

    y_dist = pylops_mpi.DistributedArray(
        global_shape=n,
        base_comm=comm,
        partition=pylops_mpi.Partition.SCATTER,
        engine=backend,
        dtype=np.float64,
    )
    y_dist[:] = np.random.normal(0.0, 1.0, y_dist.local_array.shape)

    local_extent = halo_op.local_extent[0]
    DOp = pylops.FirstDerivative(
        dims=local_extent,
        axis=0,
        kind="forward",
        dtype=np.float64,
    )
    DOp_dist = pylops_mpi.MPIBlockDiag([DOp], base_comm=comm, dtype=np.float64)
    Op_dist = halo_op.H @ DOp_dist @ halo_op

    dottest(Op_dist, x_dist, y_dist, n, n)

    y_dist = Op_dist @ x_dist
    y_adj_dist = Op_dist.H @ x_dist
    y = y_dist.asarray()
    y_adj = y_adj_dist.asarray()

    x_global = x_dist.asarray()
    if rank == 0:
        DOp_serial = pylops.FirstDerivative(
            dims=n,
            axis=0,
            kind="forward",
            dtype=np.float64,
        )
        y_serial = DOp_serial @ x_global
        y_adj_serial = DOp_serial.H @ x_global
        assert_allclose(y, y_serial, rtol=1e-14)
        assert_allclose(y_adj, y_adj_serial, rtol=1e-14)
