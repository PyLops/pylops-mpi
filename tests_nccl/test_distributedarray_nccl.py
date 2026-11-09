"""Test the DistributedArray class
Designed to run with n GPUs (with 1 MPI process per GPU)
$ mpiexec -n 3 pytest test_distributedarray_nccl.py --with-mpi

This file employs the same test sets as test_distributedarray under NCCL environment
"""

import cupy as cp
import numpy as np
import pytest
from mpi4py import MPI
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
par1j = {
    "global_shape": (501, 500),
    "partition": Partition.SCATTER,
    "dtype": np.complex128,
    "axis": 0,
}
par2 = {
    "global_shape": (500, 501),
    "partition": Partition.BROADCAST,
    "dtype": np.float64,
    "axis": 1,
}
par2j = {
    "global_shape": (501, 500),
    "partition": Partition.BROADCAST,
    "dtype": np.complex128,
    "axis": 0,
}

par3 = {
    "global_shape": (200, 201, 101),
    "partition": Partition.SCATTER,
    "dtype": np.float64,
    "axis": 1,
}

par3j = {
    "global_shape": (200, 201, 101),
    "partition": Partition.SCATTER,
    "dtype": np.complex128,
    "axis": 2,
}

par4 = {
    "x": np.random.normal(100, 100, (500, 501)),
    "partition": Partition.SCATTER,
    "axis": 1,
    "norm_axis": 0,
}

par4j = {
    "x": np.random.normal(100, 100, (500, 501))
    + 1.0j * np.random.normal(50, 50, (500, 501)),
    "partition": Partition.SCATTER,
    "axis": 1,
    "norm_axis": 0,
}

par5 = {
    "x": np.random.normal(300, 300, (500, 501)),
    "partition": Partition.SCATTER,
    "axis": 1,
    "norm_axis": 1,
}

par5j = {
    "x": np.random.normal(300, 300, (500, 501))
    + 1.0j * np.random.normal(50, 50, (500, 501)),
    "partition": Partition.SCATTER,
    "axis": 1,
    "norm_axis": 0,
}

par6 = {
    "x": np.random.normal(100, 100, (600, 600)),
    "partition": Partition.SCATTER,
    "axis": 0,
    "norm_axis": 1,
}

par7 = {
    "x": np.random.normal(300, 300, (600, 600)),
    "partition": Partition.SCATTER,
    "axis": 0,
    "norm_axis": 0,
}

par8 = {
    "x": np.random.normal(100, 100, (1200,)),
    "partition": Partition.SCATTER,
    "axis": 0,
    "norm_axis": 0,
}

par9 = {
    "x": np.random.normal(300, 300, (1200,)),
    "partition": Partition.SCATTER,
    "axis": 0,
    "norm_axis": 0,
}

# BROADCAST and UNSAFE_BROADCAST counterparts
par4b = {**par4, "partition": Partition.BROADCAST}
par5b = {**par5, "partition": Partition.BROADCAST}
par6b = {**par6, "partition": Partition.BROADCAST}
par7b = {**par7, "partition": Partition.BROADCAST}
par8b = {**par8, "partition": Partition.BROADCAST}
par9b = {**par9, "partition": Partition.BROADCAST}
par4jb = {**par4j, "partition": Partition.BROADCAST}
par5jb = {**par5j, "partition": Partition.BROADCAST}
par4u = {**par4, "partition": Partition.UNSAFE_BROADCAST}
par5u = {**par5, "partition": Partition.UNSAFE_BROADCAST}
par4ju = {**par4j, "partition": Partition.UNSAFE_BROADCAST}
par5ju = {**par5j, "partition": Partition.UNSAFE_BROADCAST}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j), (par3), (par3j)])
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
    # Test for zeros_like (source is left untouched)
    zeros_like = distributed_ones.zeros_like()
    assert isinstance(zeros_like, DistributedArray)
    assert_allclose(
        zeros_like.local_array.get(),
        np.zeros(shape=distributed_ones.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )
    assert_allclose(
        zeros_like.asarray().get(),
        np.zeros(shape=distributed_ones.global_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )
    assert_allclose(
        distributed_ones.local_array.get(),
        np.ones(shape=distributed_ones.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )
    # Test for empty_like (uninitialized values, separate buffer)
    empty_like = distributed_ones.empty_like()
    assert isinstance(empty_like, DistributedArray)
    assert empty_like.global_shape == distributed_ones.global_shape
    assert empty_like.local_shape == distributed_ones.local_shape
    assert empty_like.dtype == distributed_ones.dtype
    assert empty_like.partition == distributed_ones.partition
    assert empty_like.axis == distributed_ones.axis
    empty_like[:] = 0
    assert_allclose(
        distributed_ones.local_array.get(),
        np.ones(shape=distributed_ones.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )
    # Test for copy (same values, separate buffer)
    copy = distributed_ones.copy()
    assert isinstance(copy, DistributedArray)
    assert_allclose(
        copy.local_array.get(), distributed_ones.local_array.get(), rtol=1e-14
    )
    copy[:] = 0
    assert_allclose(
        distributed_ones.local_array.get(),
        np.ones(shape=distributed_ones.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )


@pytest.mark.mpi(min_size=2)
def test_distributed_like_checks_nccl():
    """Check behaviour of _like"""
    x = DistributedArray(
        global_shape=(4 * MPI.COMM_WORLD.Get_size(), 3),
        base_comm_nccl=nccl_comm,
        dtype=np.float64,
        engine="cupy",
    )

    # dtype of local_array wins over dtype
    y = x._like(
        local_array=cp.zeros(x.local_shape, dtype=np.complex64), dtype=np.float64
    )
    assert y.dtype == np.complex64

    # dtype is used when allocating
    assert x._like(dtype=np.float32).local_array.dtype == np.float32
    assert x._like().local_array.dtype == np.float64

    # check error is raised if local_array shape does not match
    # that of the input distributed array
    with pytest.raises(ValueError, match="expected"):
        x._like(local_array=cp.zeros((x.local_shape[0] + 1, 3)))


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("masked", [False, True])
def test_distributed_like_no_collectives_nccl(masked):
    """Check that the output DistributedArray of a basic operation
    reuses communicator and cached local shapes"""
    import sys

    DA = sys.modules["pylops_mpi.DistributedArray"]
    size = MPI.COMM_WORLD.Get_size()
    mask = None
    if masked:
        if size % 2 != 0:
            pytest.skip("Number of processes is not divisible by 2")
        mask = np.repeat(np.arange(2), size // 2)
    x = DistributedArray(
        global_shape=(4 * size, 3),
        base_comm_nccl=nccl_comm,
        mask=mask,
        engine="cupy",
    )
    x[:] = 1.0
    _ = x.local_shapes

    orig, nsplit = DA.subcomm_split, [0]

    def counting(*args, **kwargs):
        nsplit[0] += 1
        return orig(*args, **kwargs)

    DA.subcomm_split = counting
    try:
        for arr in [
            x + x,
            x - x,
            x * x,
            x * 2.0,
            -x,
            x.copy(),
            x.conj(),
            x.zeros_like(),
            x.empty_like(),
        ]:
            assert arr.sub_comm is x.sub_comm
            if not masked:
                assert arr.sub_comm is arr.base_comm_nccl
            assert arr.mask is x.mask
            assert arr.local_shapes is x.local_shapes
            assert arr.local_shape == x.local_shape
    finally:
        DA.subcomm_split = orig
    assert nsplit[0] == 0


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par4), (par4j), (par5), (par5j)])
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
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j), (par3), (par3j)])
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
@pytest.mark.parametrize(
    "par", [(par4), (par4j), (par5), (par5j), (par6), (par6b), (par7), (par7b)]
)
def test_redistribute(par):
    x_gpu = cp.asarray(par["x"])
    dist_array = DistributedArray.to_dist(
        x=x_gpu,
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        axis=par["axis"],
    )
    redist_array = dist_array.redistribute(axis=par["axis"] - 1)
    assert isinstance(redist_array, DistributedArray)
    assert_allclose(
        dist_array.asarray().get(), redist_array.asarray().get(), rtol=1e-13
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par1, par2", [(par4, par5), (par4j, par5j)])
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
    "par1, par2",
    [(par4b, par5b), (par4jb, par5jb), (par4u, par5u), (par4ju, par5ju)],
)
def test_distributed_broadcast_math_nccl(par1, par2):
    """Test the Element-Wise Addition, Subtraction and Multiplication
    for BROADCAST and UNSAFE_BROADCAST arrays"""
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
@pytest.mark.parametrize("par1, par2", [(par4, par5), (par4j, par5j)])
def test_distributed_out_nccl(par1, par2):
    """Test arithmetic with out= and in-place operators"""
    arr1 = DistributedArray.to_dist(x=cp.asarray(par1["x"]), base_comm_nccl=nccl_comm)
    arr2 = DistributedArray.to_dist(x=cp.asarray(par2["x"]), base_comm_nccl=nccl_comm)

    for name, op in [
        ("add", np.add),
        ("subtract", np.subtract),
        ("multiply", np.multiply),
    ]:
        # Benchmark, writing output to fresh DistributedArray
        expected = op(par1["x"], par2["x"])

        # Out as empty from arr1
        out = arr1.empty_like()

        # Ops with 2 DistributedArray writing to
        # pre-instantiated DistributedArray
        res = getattr(arr1, name)(arr2, out=out)
        assert res is out
        assert_allclose(out.asarray().get(), expected, rtol=1e-14)

        # Ops with DistributedArray and scalar writing to
        # pre-instantiated DistributedArray
        out = arr1.empty_like()
        getattr(arr1, name)(2.5, out=out)
        assert_allclose(out.asarray().get(), op(par1["x"], 2.5), rtol=1e-14)

        # Check input operands are untouched
        assert_allclose(arr1.asarray().get(), par1["x"], rtol=1e-14)
        assert_allclose(arr2.asarray().get(), par2["x"], rtol=1e-14)

    # Out acting as second operand as well as output
    out = arr2.copy()
    arr1.subtract(out, out=out)
    assert_allclose(out.asarray().get(), par1["x"] - par2["x"], rtol=1e-14)

    # Negation
    assert_allclose((-arr1).asarray().get(), -par1["x"], rtol=1e-14)

    # In-place operators: check correctness and that they keep the storage
    for opname, op in [
        ("__iadd__", np.add),
        ("__isub__", np.subtract),
        ("__imul__", np.multiply),
    ]:
        for other, other_g in [(arr2, par2["x"]), (2.5, 2.5)]:
            y = arr1.copy()
            local = y.local_array
            res = getattr(y, opname)(other)
            assert res is y
            assert y.local_array is local
            assert_allclose(y.asarray().get(), op(par1["x"], other_g), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par1, par2",
    [(par4b, par5b), (par4jb, par5jb), (par4u, par5u), (par4ju, par5ju)],
)
def test_distributed_broadcast_out_nccl(par1, par2):
    """Test arithmetic with out= and in-place operators for BROADCAST and
    UNSAFE_BROADCAST arrays"""
    arr1 = DistributedArray.to_dist(
        x=cp.asarray(par1["x"]),
        base_comm_nccl=nccl_comm,
        partition=par1["partition"],
    )
    arr2 = DistributedArray.to_dist(
        x=cp.asarray(par2["x"]),
        base_comm_nccl=nccl_comm,
        partition=par2["partition"],
    )

    for name, op in [
        ("add", np.add),
        ("subtract", np.subtract),
        ("multiply", np.multiply),
    ]:
        # Benchmark, writing output to fresh DistributedArray
        expected = op(par1["x"], par2["x"])

        # Out as empty from arr1
        out = arr1.empty_like()

        # Ops with 2 DistributedArray writing to
        # pre-instantiated DistributedArray
        res = getattr(arr1, name)(arr2, out=out)
        assert res is out
        assert_allclose(out.asarray().get(), expected, rtol=1e-14)

        # Ops with DistributedArray and scalar writing to
        # pre-instantiated DistributedArray
        out = arr1.empty_like()
        getattr(arr1, name)(2.5, out=out)
        assert_allclose(out.asarray().get(), op(par1["x"], 2.5), rtol=1e-14)

        # Check input operands are untouched
        assert_allclose(arr1.asarray().get(), par1["x"], rtol=1e-14)
        assert_allclose(arr2.asarray().get(), par2["x"], rtol=1e-14)

    # Out acting as second operand as well as output
    out = arr2.copy()
    arr1.subtract(out, out=out)
    assert_allclose(out.asarray().get(), par1["x"] - par2["x"], rtol=1e-14)

    # Negation
    assert_allclose((-arr1).asarray().get(), -par1["x"], rtol=1e-14)

    # In-place operators: check correctness and that they keep the storage
    for opname, op in [
        ("__iadd__", np.add),
        ("__isub__", np.subtract),
        ("__imul__", np.multiply),
    ]:
        for other, other_g in [(arr2, par2["x"]), (2.5, 2.5)]:
            y = arr1.copy()
            local = y.local_array
            res = getattr(y, opname)(other)
            assert res is y
            assert y.local_array is local
            assert_allclose(y.asarray().get(), op(par1["x"], other_g), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par1, par2", [(par6, par7), (par6b, par7b), (par8, par9), (par8b, par9b)]
)
def test_distributed_dot_nccl(par1, par2):
    """Test Distributed Dot product"""
    x1_gpu = cp.asarray(par1["x"])
    x2_gpu = cp.asarray(par2["x"])
    arr1 = DistributedArray.to_dist(
        x=x1_gpu,
        base_comm_nccl=nccl_comm,
        partition=par1["partition"],
        axis=par1["axis"],
    )
    arr2 = DistributedArray.to_dist(
        x=x2_gpu,
        base_comm_nccl=nccl_comm,
        partition=par2["partition"],
        axis=par2["axis"],
    )
    assert_allclose(
        arr1.dot(arr2),
        np.dot(par1["x"].flatten(), par2["x"].flatten()),
        rtol=1e-14,
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par",
    [
        (par4),
        (par4j),
        (par5),
        (par5j),
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
        arr.norm(ord=1, axis=par["norm_axis"]).get(),
        np.linalg.norm(par["x"], ord=1, axis=par["norm_axis"]),
        rtol=1e-14,
    )
    assert_allclose(
        arr.norm(ord=np.inf, axis=par["norm_axis"]).get(),
        np.linalg.norm(par["x"], ord=np.inf, axis=par["norm_axis"]),
        rtol=1e-14,
    )
    assert_allclose(arr.norm(), np.linalg.norm(par["x"].flatten()), rtol=1e-13)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par4), (par4j)])
def test_distributed_norm_dot_views_nccl(par):
    """Check 2-norm and dot do not modify (or depend on copies of) local arrays"""
    x = DistributedArray.to_dist(x=cp.asarray(par["x"]), base_comm_nccl=nccl_comm)
    x_before = x.local_array.copy()
    assert_allclose(x.norm(), np.linalg.norm(par["x"].ravel()), rtol=1e-12)
    assert_allclose(x.dot(x, vdot=True), np.vdot(par["x"], par["x"]), rtol=1e-12)
    assert_allclose(x.local_array.get(), x_before.get())


@pytest.mark.mpi(min_size=2)
def test_distributed_norm_overflow_nccl():
    """Check 2-norm of overflowing complex data is inf (not nan)"""
    x = DistributedArray(
        global_shape=(4 * MPI.COMM_WORLD.Get_size(),),
        base_comm_nccl=nccl_comm,
        dtype=np.complex128,
        engine="cupy",
    )
    x[:] = (1.0 + 1.0j) * 1e160
    assert np.isinf(x.norm())


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
    x_gpu = cp.asarray(par["x"])
    if par["axis"] != 0:
        x_gpu = cp.swapaxes(x_gpu, par["axis"], 0)
    for isub in range(1, nsub):
        x_gpu[
            (x_gpu.shape[0] // nsub) * isub : (x_gpu.shape[0] // nsub) * (isub + 1)
        ] = x_gpu[: x_gpu.shape[0] // nsub]
    if par["axis"] != 0:
        x_gpu = np.swapaxes(x_gpu, 0, par["axis"])

    arr = DistributedArray.to_dist(
        x=x_gpu,
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        mask=mask,
        axis=par["axis"],
    )

    # Global view
    xloc = arr.asarray()
    assert xloc.shape == x_gpu.shape

    # Global masked view
    xmaskedloc = arr.asarray(masked=True)
    xmasked_shape = list(x_gpu.shape)
    xmasked_shape[par["axis"]] = int(xmasked_shape[par["axis"]] // nsub)
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
        arr1.dot(arr2), np.dot(x1.flatten(), x2.flatten()) / nsub, rtol=1e-14
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
        arr.norm(ord=1, axis=par["norm_axis"]).get(),
        np.linalg.norm(par["x"], ord=1, axis=par["norm_axis"])
        / (nsub if par["axis"] == par["norm_axis"] else 1),
        rtol=1e-14,
    )
    assert_allclose(
        arr.norm(ord=2, axis=par["norm_axis"]).get(),
        np.linalg.norm(par["x"], ord=2, axis=par["norm_axis"])
        / (np.sqrt(nsub) if par["axis"] == par["norm_axis"] else 1),
        rtol=1e-13,
    )
    assert_allclose(
        arr.norm(ord=np.inf, axis=par["norm_axis"]).get(),
        np.linalg.norm(par["x"], ord=np.inf, axis=par["norm_axis"]),
        rtol=1e-14,
    )
