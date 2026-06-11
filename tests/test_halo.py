"""Test the the Halo class
    Designed to run with n processes
    $ mpiexec -n 10 pytest test_halo.py --with-mpi
"""
import os
import math

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
import pylops_mpi
from pylops_mpi.basicoperators import MPIHalo, halo_block_split
from pylops_mpi.utils.dottest import dottest

np.random.seed(42)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if backend == "cupy":
    device_id = rank % np.cuda.runtime.getDeviceCount()
    np.cuda.Device(device_id).use()


par1 = {"dims": (16 * size,), "proc_grid_shape": (size,)}
par2 = {"dims": (16 * size, 8), "proc_grid_shape": (size, 1)}
par3 = {"dims": (8, 16 * size), "proc_grid_shape": (1, size)}
par4 = {"dims": (16 * size, 8, 8), "proc_grid_shape": (size, 1, 1)}
par5 = {"dims": (8, 16 * size, 8), "proc_grid_shape": (1, size, 1)}
par6 = {"dims": (8, 8, 16 * size), "proc_grid_shape": (1, 1, size)}

uneven_par1 = {"dims": (16 * size + 1,), "proc_grid_shape": (size,)}
uneven_par2 = {"dims": (16 * size + 1, 8), "proc_grid_shape": (size, 1)}
uneven_par3 = {"dims": (8, 16 * size + 1), "proc_grid_shape": (1, size)}
uneven_par4 = {"dims": (8, 8, 16 * size + 1), "proc_grid_shape": (1, 1, size)}


def _expected_haloed_block(global_array, dims, proc_grid_shape, halo):
    slices = halo_block_split(dims, comm, proc_grid_shape)
    local_dims = tuple(
        (dims[ax] if sl.stop is None else sl.stop) - sl.start
        for ax, sl in enumerate(slices)
    )
    expected_shape = tuple(
        local_dims[ax] + halo[2 * ax] + halo[2 * ax + 1]
        for ax in range(len(local_dims))
    )
    expected = np.zeros(expected_shape, dtype=np.float64)

    source_slices = []
    target_slices = []
    for ax, sl in enumerate(slices):
        local_start = sl.start
        local_stop = dims[ax] if sl.stop is None else sl.stop
        before, after = halo[2 * ax], halo[2 * ax + 1]

        source_start = max(0, local_start - before)
        source_stop = min(dims[ax], local_stop + after)
        target_start = before - (local_start - source_start)
        target_stop = target_start + (source_stop - source_start)

        source_slices.append(slice(source_start, source_stop))
        target_slices.append(slice(target_start, target_stop))

    expected[tuple(target_slices)] = global_array[tuple(source_slices)]

    return expected


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
@pytest.mark.parametrize("halo_kind", ["scalar", "ndim_tuple", "per_side_tuple"])
def test_halo(par, halo_kind):
    dims, proc_grid_shape = par["dims"], par["proc_grid_shape"]
    ndim = len(dims)

    if halo_kind == "scalar":
        halo = 1
        cart = comm.Create_cart(proc_grid_shape, periods=[False] * ndim, reorder=True)
        trimmed = [halo] * (2 * ndim)
        for ax in range(ndim):
            before, after = cart.Shift(ax, 1)
            if before == MPI.PROC_NULL:
                trimmed[2 * ax] = 0
            if after == MPI.PROC_NULL:
                trimmed[2 * ax + 1] = 0
        expected_halo = tuple(trimmed)
    elif halo_kind == "ndim_tuple":
        halo = tuple(range(1, ndim + 1))
        expected_halo = tuple(value for value in halo for _ in range(2))
    else:
        halo = tuple(value for value in range(1, ndim + 1) for _ in range(2))
        expected_halo = halo

    halo_op = MPIHalo(
        dims=dims,
        halo=halo,
        proc_grid_shape=proc_grid_shape,
        comm=comm,
        dtype=np.float64,
    )

    global_array = np.arange(math.prod(dims), dtype=np.float64).reshape(dims)
    local_block = global_array[halo_block_split(dims, comm, proc_grid_shape)]
    expected_block = _expected_haloed_block(
        global_array,
        dims,
        proc_grid_shape,
        expected_halo,
    )

    x_dist = pylops_mpi.DistributedArray(
        global_shape=math.prod(dims),
        base_comm=comm,
        partition=pylops_mpi.Partition.SCATTER,
        engine=backend,
        dtype=np.float64,
    )
    x_dist[:] = local_block.ravel()

    y_dist = halo_op @ x_dist
    assert y_dist.local_array.shape == (expected_block.size,)
    assert y_dist.global_shape == (sum(comm.allgather(expected_block.size)),)
    assert_allclose(y_dist.local_array, expected_block.ravel(), rtol=1e-14)

    x_adj_dist = halo_op.H @ y_dist
    assert x_adj_dist.global_shape == (math.prod(dims),)
    assert_allclose(x_adj_dist.local_array, local_block.ravel(), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(uneven_par1), (uneven_par2), (uneven_par3), (uneven_par4)])
def test_halo_uneven_global_size(par):
    dims, proc_grid_shape = par["dims"], par["proc_grid_shape"]
    ndim = len(dims)
    halo = 1
    cart = comm.Create_cart(proc_grid_shape, periods=[False] * ndim, reorder=True)
    expected_halo = [halo] * (2 * ndim)
    for ax in range(ndim):
        before, after = cart.Shift(ax, 1)
        if before == MPI.PROC_NULL:
            expected_halo[2 * ax] = 0
        if after == MPI.PROC_NULL:
            expected_halo[2 * ax + 1] = 0

    halo_op = MPIHalo(
        dims=dims,
        halo=halo,
        proc_grid_shape=proc_grid_shape,
        comm=comm,
        dtype=np.float64,
    )

    global_array = np.arange(math.prod(dims), dtype=np.float64).reshape(dims)
    local_block = global_array[halo_block_split(dims, comm, proc_grid_shape)]
    expected_block = _expected_haloed_block(
        global_array,
        dims,
        proc_grid_shape,
        tuple(expected_halo),
    )
    local_shapes = comm.allgather(local_block.size)

    x_dist = pylops_mpi.DistributedArray(
        global_shape=math.prod(dims),
        base_comm=comm,
        partition=pylops_mpi.Partition.SCATTER,
        local_shapes=local_shapes,
        engine=backend,
        dtype=np.float64,
    )
    x_dist[:] = local_block.ravel()

    y_dist = halo_op @ x_dist
    assert y_dist.local_array.shape == (expected_block.size,)
    assert_allclose(y_dist.local_array, expected_block.ravel(), rtol=1e-14)

    x_adj_dist = halo_op.H @ y_dist
    assert x_adj_dist.local_array.shape == (local_block.size,)
    assert_allclose(x_adj_dist.local_array, local_block.ravel(), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_halo_tuple_boundary_zeros_match_scalar(par):
    dims, proc_grid_shape = par["dims"], par["proc_grid_shape"]
    ndim = len(dims)

    cart = comm.Create_cart(proc_grid_shape, periods=[False] * ndim, reorder=True)
    tuple_halo = [1] * (2 * ndim)
    for ax in range(ndim):
        before, after = cart.Shift(ax, 1)
        if before == MPI.PROC_NULL:
            tuple_halo[2 * ax] = 0
        if after == MPI.PROC_NULL:
            tuple_halo[2 * ax + 1] = 0

    scalar_halo_op = MPIHalo(
        dims=dims,
        halo=1,
        proc_grid_shape=proc_grid_shape,
        comm=comm,
        dtype=np.float64,
    )
    tuple_halo_op = MPIHalo(
        dims=dims,
        halo=tuple(tuple_halo),
        proc_grid_shape=proc_grid_shape,
        comm=comm,
        dtype=np.float64,
    )

    global_array = np.arange(math.prod(dims), dtype=np.float64).reshape(dims)
    local_block = global_array[halo_block_split(dims, comm, proc_grid_shape)]
    x_dist = pylops_mpi.DistributedArray(
        global_shape=math.prod(dims),
        base_comm=comm,
        partition=pylops_mpi.Partition.SCATTER,
        engine=backend,
        dtype=np.float64,
    )
    x_dist[:] = local_block.ravel()

    y_scalar_dist = scalar_halo_op @ x_dist
    y_tuple_dist = tuple_halo_op @ x_dist
    assert y_tuple_dist.global_shape == y_scalar_dist.global_shape
    assert y_tuple_dist.local_array.shape == y_scalar_dist.local_array.shape
    assert_allclose(y_tuple_dist.local_array, y_scalar_dist.local_array, rtol=1e-14)

    x_scalar_adj_dist = scalar_halo_op.H @ y_scalar_dist
    x_tuple_adj_dist = tuple_halo_op.H @ y_tuple_dist
    assert_allclose(
        x_tuple_adj_dist.local_array,
        x_scalar_adj_dist.local_array,
        rtol=1e-14,
    )


@pytest.mark.mpi(min_size=2)
def test_halo_invalid_grid_shape():
    with pytest.raises(ValueError, match="does not match comm size"):
        MPIHalo(
            dims=(16 * size,),
            halo=1,
            proc_grid_shape=(size + 1,),
            comm=comm,
            dtype=np.float64,
        )


@pytest.mark.mpi(min_size=2)
def test_halo_invalid_halo_shape():
    with pytest.raises(ValueError, match="Invalid halo length"):
        MPIHalo(
            dims=(16 * size,),
            halo=(1, 2, 3),
            proc_grid_shape=(size,),
            comm=comm,
            dtype=np.float64,
        )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("halo", [-1, (-1,), (1, -1)])
def test_halo_invalid_negative_halo(halo):
    with pytest.raises(ValueError, match="non-negative"):
        MPIHalo(
            dims=(16 * size,),
            halo=halo,
            proc_grid_shape=(size,),
            comm=comm,
            dtype=np.float64,
        )


@pytest.mark.mpi(min_size=2)
def test_halo_rejects_halo_wider_than_local_block():
    dims = (4 * size,)
    with pytest.raises(ValueError, match="exceeds local block size"):
        MPIHalo(
            dims=dims,
            halo=5,
            proc_grid_shape=(size,),
            comm=comm,
            dtype=np.float64,
        )


@pytest.mark.mpi(min_size=2)
def test_halo_invalid_asymmetric_distributed_halo():
    dims = (16 * size,)
    with pytest.raises(ValueError, match="does not match neighbor"):
        MPIHalo(
            dims=dims,
            halo=(1, 2),
            proc_grid_shape=(size,),
            comm=comm,
            dtype=np.float64,
        )


@pytest.mark.mpi(min_size=2)
def test_halo_rejects_broadcast_input():
    dims = (16 * size,)
    halo_op = MPIHalo(
        dims=dims,
        halo=1,
        proc_grid_shape=(size,),
        comm=comm,
        dtype=np.float64,
    )
    x_dist = pylops_mpi.DistributedArray(
        global_shape=math.prod(dims),
        base_comm=comm,
        partition=pylops_mpi.Partition.BROADCAST,
        engine=backend,
        dtype=np.float64,
    )
    y_dist = pylops_mpi.DistributedArray(
        global_shape=halo_op.shape[0],
        base_comm=comm,
        partition=pylops_mpi.Partition.BROADCAST,
        engine=backend,
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match=f"{pylops_mpi.Partition.SCATTER}"):
        halo_op @ x_dist
    with pytest.raises(ValueError, match=f"{pylops_mpi.Partition.SCATTER}"):
        halo_op.H @ y_dist


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_halo_first_derivative(par):
    dims, proc_grid_shape = par["dims"], par["proc_grid_shape"]
    axis = proc_grid_shape.index(size)
    n = math.prod(dims)
    halo = 1

    halo_op = MPIHalo(
        dims=dims,
        halo=halo,
        proc_grid_shape=proc_grid_shape,
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

    y_dist = pylops_mpi.DistributedArray(
        global_shape=n,
        base_comm=comm,
        partition=pylops_mpi.Partition.SCATTER,
        engine=backend,
        dtype=np.float64,
    )

    model_global = np.arange(n, dtype=np.float64).reshape(dims)
    data_global = (np.arange(n, dtype=np.float64).reshape(dims) + 1.0) / n
    local_slices = halo_block_split(dims, comm, proc_grid_shape)
    x_dist[:] = model_global[local_slices].ravel()
    y_dist[:] = data_global[local_slices].ravel()

    cart = comm.Create_cart(proc_grid_shape, periods=[False] * len(dims), reorder=True)
    expected_halo = [halo] * (2 * len(dims))
    for ax in range(len(dims)):
        before, after = cart.Shift(ax, 1)
        if before == MPI.PROC_NULL: expected_halo[2 * ax]    = 0
        if after == MPI.PROC_NULL: expected_halo[2 * ax + 1] = 0

    local_extent = _expected_haloed_block(
        np.zeros(dims, dtype=np.float64),
        dims,
        proc_grid_shape,
        tuple(expected_halo),
    ).shape

    DOp = pylops.FirstDerivative(
        dims=local_extent,
        axis=axis,
        kind="forward",
        dtype=np.float64,
    )
    DOp_dist = pylops_mpi.MPIBlockDiag([DOp], base_comm=comm, dtype=np.float64)
    Op_dist = halo_op.H @ DOp_dist @ halo_op

    dottest(Op_dist, x_dist, y_dist, n, n)

    y_dist = Op_dist @ x_dist
    y_adj_dist = Op_dist.H @ x_dist

    DOp_serial = pylops.FirstDerivative(
        dims=dims,
        axis=axis,
        kind="forward",
        dtype=np.float64,
    )
    y_serial = (DOp_serial @ model_global.ravel()).reshape(dims)
    y_adj_serial = (DOp_serial.H @ model_global.ravel()).reshape(dims)
    assert_allclose(y_dist.local_array, y_serial[local_slices].ravel(), rtol=1e-14)
    assert_allclose(y_adj_dist.local_array, y_adj_serial[local_slices].ravel(), rtol=1e-14)
