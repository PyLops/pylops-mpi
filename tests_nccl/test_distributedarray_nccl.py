"""Test the DistributedArray class
Designed to run with n GPUs (with 1 MPI process per GPU)
$ mpiexec -n 3 pytest test_distributedarray_nccl.py --with-mpi

This file employs the same test sets as test_distributedarray under NCCL environment
"""

import numpy as np
import cupy as cp
from mpi4py import MPI
import pytest
from numpy.testing import assert_allclose

from pylops_mpi import DistributedArray, Partition
from pylops_mpi.DistributedArray import local_split
from pylops_mpi.utils._nccl import initialize_nccl_comm

np.random.seed(42)

nccl_comm = initialize_nccl_comm()

par1 = {
    "global_shape": (500, 501),
    "partition": Partition.SCATTER,
    "dtype": np.float64,
    "axis": 1,
}

par2 = {
    "global_shape": (500, 501),
    "partition": Partition.BROADCAST,
    "dtype": np.float64,
    "axis": 1,
}

par3 = {
    "global_shape": (200, 201, 101),
    "partition": Partition.SCATTER,
    "dtype": np.float64,
    "axis": 1,
}

par4 = {
    "x": np.random.normal(100, 100, (500, 501)),
    "partition": Partition.SCATTER,
    "axis": 1,
}

par5 = {
    "x": np.random.normal(300, 300, (500, 501)),
    "partition": Partition.SCATTER,
    "axis": 1,
}

par6 = {
    "x": np.random.normal(100, 100, (600, 600)),
    "partition": Partition.SCATTER,
    "axis": 0,
}

par6b = {
    "x": np.random.normal(100, 100, (600, 600)),
    "partition": Partition.BROADCAST,
    "axis": 0,
}

par7 = {
    "x": np.random.normal(300, 300, (600, 600)),
    "partition": Partition.SCATTER,
    "axis": 0,
}

par7b = {
    "x": np.random.normal(300, 300, (600, 600)),
    "partition": Partition.BROADCAST,
    "axis": 0,
}

par8 = {
    "x": np.random.normal(100, 100, (1200,)),
    "partition": Partition.SCATTER,
    "axis": 0,
}

par8b = {
    "x": np.random.normal(100, 100, (1200,)),
    "partition": Partition.BROADCAST,
    "axis": 0,
}

par9 = {
    "x": np.random.normal(300, 300, (1200,)),
    "partition": Partition.SCATTER,
    "axis": 0,
}

par9b = {
    "x": np.random.normal(300, 300, (1200,)),
    "partition": Partition.BROADCAST,
    "axis": 0,
}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_creation_nccl(par):
    """Test creation of local arrays"""
    distributed_array = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    loc_shape = local_split(
        distributed_array.global_shape,
        distributed_array.base_comm,
        distributed_array.partition,
        distributed_array.axis,
    )
    assert distributed_array.global_shape == par["global_shape"]
    assert distributed_array.local_shape == loc_shape
    assert isinstance(distributed_array, DistributedArray)
    # Distributed array of ones
    distributed_ones = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_ones[:] = 1
    # Distributed array of zeroes
    distributed_zeroes = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_zeroes[:] = 0
    # Test for distributed ones
    assert isinstance(distributed_ones, DistributedArray)
    assert_allclose(
        distributed_ones.local_array.get(),
        np.ones(shape=distributed_ones.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )
    assert_allclose(
        distributed_ones.asarray().get(),
        np.ones(shape=distributed_ones.global_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )
    # Test for distributed zeroes
    assert isinstance(distributed_zeroes, DistributedArray)
    assert_allclose(
        distributed_zeroes.local_array.get(),
        np.zeros(shape=distributed_zeroes.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )
    assert_allclose(
        distributed_zeroes.asarray().get(),
        np.zeros(shape=distributed_zeroes.global_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par4), (par5)])
def test_to_dist_nccl(par):
    """Test the ``to_dist`` method"""
    x_gpu = cp.asarray(par["x"])
    dist_array = DistributedArray.to_dist(
        x=x_gpu,
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        axis=par["axis"],
    )
    assert isinstance(dist_array, DistributedArray)
    assert dist_array.global_shape == par["x"].shape
    assert dist_array.axis == par["axis"]


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_local_shapes_nccl(par):
    """Test the `local_shapes` parameter in DistributedArray"""
    # Reverse the local_shapes to test the local_shapes parameter
    loc_shapes = MPI.COMM_WORLD.allgather(
        local_split(par["global_shape"], MPI.COMM_WORLD, par["partition"], par["axis"])
    )[::-1]
    distributed_array = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        axis=par["axis"],
        local_shapes=loc_shapes,
        dtype=par["dtype"],
        engine="cupy",
    )
    assert isinstance(distributed_array, DistributedArray)
    assert distributed_array.local_shape == loc_shapes[distributed_array.rank]

    # Distributed ones
    distributed_array[:] = 1
    assert_allclose(
        distributed_array.local_array.get(),
        np.ones(loc_shapes[distributed_array.rank], dtype=par["dtype"]),
        rtol=1e-14,
    )
    assert_allclose(
        distributed_array.asarray().get(),
        np.ones(par["global_shape"], dtype=par["dtype"]),
        rtol=1e-14,
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par1, par2", [(par4, par5)])
def test_distributed_math_nccl(par1, par2):
    """Test the Element-Wise Addition, Subtraction and Multiplication"""
    x1_gpu = cp.asarray(par1["x"])
    x2_gpu = cp.asarray(par2["x"])
    arr1 = DistributedArray.to_dist(
        x=x1_gpu, base_comm_nccl=nccl_comm, partition=par1["partition"]
    )
    arr2 = DistributedArray.to_dist(
        x=x2_gpu, base_comm_nccl=nccl_comm, partition=par2["partition"]
    )

    # Addition
    sum_array = arr1 + arr2
    assert isinstance(sum_array, DistributedArray)
    # Subtraction
    sub_array = arr1 - arr2
    assert isinstance(sub_array, DistributedArray)
    # Multiplication
    mult_array = arr1 * arr2
    assert isinstance(mult_array, DistributedArray)
    # Global array of Sum with np.add

    assert_allclose(sum_array.asarray().get(), np.add(par1["x"], par2["x"]), rtol=1e-14)
    # Global array of Subtract with np.subtract
    assert_allclose(
        sub_array.asarray().get(), np.subtract(par1["x"], par2["x"]), rtol=1e-14
    )
    # Global array of Multiplication with np.multiply
    assert_allclose(
        mult_array.asarray().get(), np.multiply(par1["x"], par2["x"]), rtol=1e-14
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par1, par2", [(par6, par7), (par6b, par7b), (par8, par9), (par8b, par9b)]
)
def test_distributed_dot_nccl(par1, par2):
    """Test Distributed Dot product"""
    x1_gpu = cp.asarray(par1["x"])
    x2_gpu = cp.asarray(par2["x"])
    arr1 = DistributedArray.to_dist(
        x=x1_gpu, base_comm_nccl=nccl_comm, partition=par1["partition"], axis=par1["axis"]
    )
    arr2 = DistributedArray.to_dist(
        x=x2_gpu, base_comm_nccl=nccl_comm, partition=par2["partition"], axis=par2["axis"]
    )
    assert_allclose(
        (arr1.dot(arr2)).get(),
        np.dot(par1["x"].flatten(), par2["x"].flatten()),
        rtol=1e-14,
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par",
    [
        (par4),
        (par5),
        (par6),
        (par6b),
        (par7),
        (par7b),
        (par8),
        (par8b),
        (par9),
        (par9b),
    ],
)
def test_distributed_norm_nccl(par):
    """Test Distributed numpy.linalg.norm method"""
    x_gpu = cp.asarray(par["x"])
    arr = DistributedArray.to_dist(x=x_gpu, base_comm_nccl=nccl_comm, axis=par["axis"])
    assert_allclose(
        arr.norm(ord=1, axis=par["axis"]).get(),
        np.linalg.norm(par["x"], ord=1, axis=par["axis"]),
        rtol=1e-14,
    )
    assert_allclose(
        arr.norm(ord=np.inf, axis=par["axis"]).get(),
        np.linalg.norm(par["x"], ord=np.inf, axis=par["axis"]),
        rtol=1e-14,
    )
    assert_allclose(arr.norm().get(), np.linalg.norm(par["x"].flatten()), rtol=1e-13)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par6), (par8)])
def test_distributed_masked_nccl(par):
    """Test Asarray with masked array"""
    # Number of subcommunicators
    if MPI.COMM_WORLD.Get_size() % 2 == 0:
        nsub = 2
    elif MPI.COMM_WORLD.Get_size() % 3 == 0:
        nsub = 3
    else:
        pass
    subsize = max(1, MPI.COMM_WORLD.Get_size() // nsub)
    mask = np.repeat(np.arange(nsub), subsize)

    # Replicate x as required in masked arrays
    x_gpu = cp.asarray(par['x'])
    if par['axis'] != 0:
        x_gpu = cp.swapaxes(x_gpu, par['axis'], 0)
    for isub in range(1, nsub):
        x_gpu[(x_gpu.shape[0] // nsub) * isub:(x_gpu.shape[0] // nsub) * (isub + 1)] = x_gpu[:x_gpu.shape[0] // nsub]
    if par['axis'] != 0:
        x_gpu = np.swapaxes(x_gpu, 0, par['axis'])

    arr = DistributedArray.to_dist(x=x_gpu, base_comm_nccl=nccl_comm, partition=par['partition'], mask=mask, axis=par['axis'])

    # Global view
    xloc = arr.asarray()
    assert xloc.shape == x_gpu.shape

    # Global masked view
    xmaskedloc = arr.asarray(masked=True)
    xmasked_shape = list(x_gpu.shape)
    xmasked_shape[par['axis']] = int(xmasked_shape[par['axis']] // nsub)
    assert xmaskedloc.shape == tuple(xmasked_shape)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par1, par2", [(par6, par7), (par6b, par7b), (par8, par9), (par8b, par9b)]
)
def test_distributed_maskeddot_nccl(par1, par2):
    """Test Distributed Dot product with masked array"""
    # number of subcommunicators
    if MPI.COMM_WORLD.Get_size() % 2 == 0:
        nsub = 2
    elif MPI.COMM_WORLD.Get_size() % 3 == 0:
        nsub = 3
    else:
        pass
    subsize = max(1, MPI.COMM_WORLD.Get_size() // nsub)
    mask = np.repeat(np.arange(nsub), subsize)
    # Replicate x1 and x2 as required in masked arrays
    x1, x2 = par1["x"], par2["x"]
    if par1["axis"] != 0:
        x1 = np.swapaxes(x1, par1["axis"], 0)
    for isub in range(1, nsub):
        x1[(x1.shape[0] // nsub) * isub : (x1.shape[0] // nsub) * (isub + 1)] = x1[
            : x1.shape[0] // nsub
        ]
    if par1["axis"] != 0:
        x1 = np.swapaxes(x1, 0, par1["axis"])
    if par2["axis"] != 0:
        x2 = np.swapaxes(x2, par2["axis"], 0)
    for isub in range(1, nsub):
        x2[(x2.shape[0] // nsub) * isub : (x2.shape[0] // nsub) * (isub + 1)] = x2[
            : x2.shape[0] // nsub
        ]
    if par2["axis"] != 0:
        x2 = np.swapaxes(x2, 0, par2["axis"])

    x1_gpu, x2_gpu = cp.asarray(x1), cp.asarray(x2)
    arr1 = DistributedArray.to_dist(
        x=x1_gpu,
        base_comm_nccl=nccl_comm,
        partition=par1["partition"],
        mask=mask,
        axis=par1["axis"],
    )
    arr2 = DistributedArray.to_dist(
        x=x2_gpu,
        base_comm_nccl=nccl_comm,
        partition=par2["partition"],
        mask=mask,
        axis=par2["axis"],
    )
    assert_allclose(
        arr1.dot(arr2).get(), np.dot(x1.flatten(), x2.flatten()) / nsub, rtol=1e-14
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par6), (par6b), (par7), (par7b), (par8), (par8b), (par9), (par9b)]
)
def test_distributed_maskednorm_nccl(par):
    """Test Distributed numpy.linalg.norm method with masked array"""
    # number of subcommunicators
    if MPI.COMM_WORLD.Get_size() % 2 == 0:
        nsub = 2
    elif MPI.COMM_WORLD.Get_size() % 3 == 0:
        nsub = 3
    else:
        pass
    subsize = max(1, MPI.COMM_WORLD.Get_size() // nsub)
    mask = np.repeat(np.arange(nsub), subsize)
    # Replicate x as required in masked arrays
    x = par["x"]
    if par["axis"] != 0:
        x = np.swapaxes(x, par["axis"], 0)
    for isub in range(1, nsub):
        x[(x.shape[0] // nsub) * isub : (x.shape[0] // nsub) * (isub + 1)] = x[
            : x.shape[0] // nsub
        ]
    if par["axis"] != 0:
        x = np.swapaxes(x, 0, par["axis"])

    x_gpu = cp.asarray(x)
    arr = DistributedArray.to_dist(
        x=x_gpu, base_comm_nccl=nccl_comm, mask=mask, axis=par["axis"]
    )
    assert_allclose(
        arr.norm(ord=1, axis=par["axis"]).get(),
        np.linalg.norm(par["x"], ord=1, axis=par["axis"]) / nsub,
        rtol=1e-14,
    )
    assert_allclose(
        arr.norm(ord=np.inf, axis=par["axis"]).get(),
        np.linalg.norm(par["x"], ord=np.inf, axis=par["axis"]),
        rtol=1e-14,
    )
    assert_allclose(
        arr.norm(ord=2, axis=par["axis"]).get(),
        np.linalg.norm(par["x"], ord=2, axis=par["axis"]) / np.sqrt(nsub),
        rtol=1e-13,
    )
