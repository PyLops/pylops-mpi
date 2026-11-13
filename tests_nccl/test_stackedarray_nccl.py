"""Test the StackedDistributedArray class
    Designed to run with n GPUs (with 1 MPI process per GPU)
    $ mpiexec -n 10 pytest test_stackedarray_nccl.py --with-mpi

This file employs the same test sets as test_stackedarray under NCCL environment
"""
import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pylops_mpi import DistributedArray, Partition, StackedDistributedArray
from pylops_mpi.utils._nccl import initialize_nccl_comm

nccl_comm = initialize_nccl_comm()

np.random.seed(42)

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


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j), (par3), (par3j)])
def test_creation_nccl(par):
    """Test creation of stacked distributed arrays"""
    # Create stacked array
    distributed_array0 = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array1 = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array0[:] = 0
    distributed_array1[:] = 1

    stacked_arrays = StackedDistributedArray([distributed_array0, distributed_array1])
    assert isinstance(stacked_arrays, StackedDistributedArray)
    assert stacked_arrays.global_shape == tuple(gs * 2 for gs in par["global_shape"])
    assert_allclose(
        stacked_arrays[0].local_array.get(),
        np.zeros(shape=distributed_array0.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )
    assert_allclose(
        stacked_arrays[1].local_array.get(),
        np.ones(shape=distributed_array1.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )

    # Asarray
    stacked_arrays_local = stacked_arrays.asarray().get()
    stacked_arrays_local_bench = np.hstack(
        [
            np.zeros(shape=distributed_array0.global_shape, dtype=par["dtype"]).ravel(),
            np.ones(shape=distributed_array1.global_shape, dtype=par["dtype"]).ravel(),
        ]
    )
    assert_allclose(stacked_arrays_local, stacked_arrays_local_bench, rtol=1e-14)

    # Modify array in place
    distributed_array0[:] = 2
    assert_allclose(
        stacked_arrays[0].local_array.get(),
        2 * np.ones(shape=distributed_array0.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )

    # zeros_like (source is left untouched)
    zeros_like = stacked_arrays.zeros_like()
    assert isinstance(zeros_like, StackedDistributedArray)
    assert zeros_like.global_shape == stacked_arrays.global_shape
    for zl, sa in zip(zeros_like.distarrays, stacked_arrays.distarrays):
        assert zl.local_shape == sa.local_shape
        assert_allclose(
            zl.local_array.get(),
            np.zeros(shape=sa.local_shape, dtype=par["dtype"]),
            rtol=1e-14,
        )
    assert_allclose(
        zeros_like.asarray().get(),
        np.zeros(stacked_arrays.asarray().shape, dtype=par["dtype"]),
        rtol=1e-14,
    )
    assert_allclose(
        stacked_arrays[0].local_array.get(),
        2 * np.ones(shape=distributed_array0.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )

    # empty_like (uninitialized values, separate buffers)
    empty_like = stacked_arrays.empty_like()
    assert isinstance(empty_like, StackedDistributedArray)
    assert empty_like.global_shape == stacked_arrays.global_shape
    for el, sa in zip(empty_like.distarrays, stacked_arrays.distarrays):
        assert el.global_shape == sa.global_shape
        assert el.local_shape == sa.local_shape
        assert el.dtype == sa.dtype
        assert el.partition == sa.partition
        assert el.axis == sa.axis
        el[:] = 0
    assert_allclose(
        stacked_arrays[0].local_array.get(),
        2 * np.ones(shape=distributed_array0.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )
    assert_allclose(
        stacked_arrays[1].local_array.get(),
        np.ones(shape=distributed_array1.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j), (par3), (par3j)])
def test_stacked_math_nccl(par):
    """Test the Element-Wise Addition, Subtraction and Multiplication, Dot-product, Norm"""
    distributed_array0 = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array1 = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array0[:] = 0
    distributed_array1[:] = cp.arange(np.prod(distributed_array1.local_shape)).reshape(
        distributed_array1.local_shape
    )

    stacked_array1 = StackedDistributedArray([distributed_array0, distributed_array1])
    stacked_array2 = StackedDistributedArray([distributed_array1, distributed_array0])

    # Addition
    sum_array = stacked_array1 + stacked_array2
    assert isinstance(sum_array, StackedDistributedArray)
    assert_allclose(
        sum_array.asarray().get(),
        np.add(stacked_array1.asarray().get(), stacked_array2.asarray().get()),
        rtol=1e-14,
    )
    # Subtraction
    sub_array = stacked_array1 - stacked_array2
    assert isinstance(sub_array, StackedDistributedArray)
    assert_allclose(
        sub_array.asarray().get(),
        np.subtract(stacked_array1.asarray().get(), stacked_array2.asarray().get()),
        rtol=1e-14,
    )
    # Multiplication
    mult_array = stacked_array1 * stacked_array2
    assert isinstance(mult_array, StackedDistributedArray)
    assert_allclose(
        mult_array.asarray().get(),
        np.multiply(stacked_array1.asarray().get(), stacked_array2.asarray().get()),
        rtol=1e-14,
    )
    # Dot-product
    dot_prod = stacked_array1.dot(stacked_array2)
    assert_allclose(
        dot_prod,
        np.dot(
            stacked_array1.asarray().flatten().get(),
            stacked_array2.asarray().flatten().get(),
        ),
        rtol=1e-14,
    )
    # Norm
    l0norm = stacked_array1.norm(0)
    l1norm = stacked_array1.norm(1)
    l2norm = stacked_array1.norm(2)

    linfnorm = stacked_array1.norm(np.inf)
    assert_allclose(
        l0norm, np.linalg.norm(stacked_array1.asarray().flatten().get(), 0), rtol=1e-14
    )
    assert_allclose(
        l1norm, np.linalg.norm(stacked_array1.asarray().flatten().get(), 1), rtol=1e-14
    )
    assert_allclose(
        l2norm, np.linalg.norm(stacked_array1.asarray().get(), 2), rtol=1e-10
    )  # needed to raise it due to how partial norms are combined (with power applied)
    assert_allclose(
        linfnorm,
        np.linalg.norm(stacked_array1.asarray().flatten().get(), np.inf),
        rtol=1e-14,
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j), (par3), (par3j)])
def test_creation_nested(par):
    """Test creation of nested stacked distributed arrays"""
    # Create stacked array
    distributed_array0 = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array1 = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array2 = DistributedArray(
        global_shape=[gs * 2 for gs in par["global_shape"]],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array0[:] = 0
    distributed_array1[:] = 1
    distributed_array2[:] = 2

    stacked_arrays_01 = StackedDistributedArray(
        [distributed_array0, distributed_array1]
    )
    stacked_arrays = StackedDistributedArray([stacked_arrays_01, distributed_array2])
    assert isinstance(stacked_arrays, StackedDistributedArray)
    assert stacked_arrays.global_shape == tuple(gs * 4 for gs in par["global_shape"])
    assert_allclose(
        stacked_arrays[0][0].local_array.get(),
        np.zeros(shape=distributed_array0.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )
    assert_allclose(
        stacked_arrays[0][1].local_array.get(),
        np.ones(shape=distributed_array1.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )
    assert_allclose(
        stacked_arrays[1].local_array.get(),
        2 * np.ones(shape=distributed_array2.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )

    # Asarray
    stacked_arrays_local = stacked_arrays.asarray().get()
    stacked_arrays_local_bench = np.hstack(
        [
            np.zeros(shape=distributed_array0.global_shape, dtype=par["dtype"]).ravel(),
            np.ones(shape=distributed_array1.global_shape, dtype=par["dtype"]).ravel(),
            2
            * np.ones(
                shape=distributed_array2.global_shape, dtype=par["dtype"]
            ).ravel(),
        ]
    )
    assert_allclose(stacked_arrays_local, stacked_arrays_local_bench, rtol=1e-14)

    # Modify array in place
    distributed_array0[:] = 2
    assert_allclose(
        stacked_arrays[0][0].local_array.get(),
        2 * np.ones(shape=distributed_array0.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )
    distributed_array2[:] = 4
    assert_allclose(
        stacked_arrays[1].local_array.get(),
        4 * np.ones(shape=distributed_array2.local_shape, dtype=par["dtype"]),
        rtol=1e-14,
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j), (par3), (par3j)])
def test_stacked_nested_math(par):
    """Test Element-Wise Addition, Subtraction and Multiplication, Dot-product, Norm
    for nested stacked distributed arrays"""
    distributed_array0 = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array1 = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array2 = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array0[:] = 0
    distributed_array1[:] = cp.arange(np.prod(distributed_array1.local_shape)).reshape(
        distributed_array1.local_shape
    )
    distributed_array2[:] = cp.ones(np.prod(distributed_array2.local_shape)).reshape(
        distributed_array1.local_shape
    )

    stacked_array_01 = StackedDistributedArray([distributed_array0, distributed_array1])
    stacked_array_12 = StackedDistributedArray([distributed_array1, distributed_array2])
    stacked_array1 = StackedDistributedArray([stacked_array_01, distributed_array2])
    stacked_array2 = StackedDistributedArray([stacked_array_12, distributed_array0])

    # Addition
    sum_array = stacked_array1 + stacked_array2
    assert isinstance(sum_array, StackedDistributedArray)
    assert_allclose(
        sum_array.asarray().get(),
        np.add(stacked_array1.asarray().get(), stacked_array2.asarray().get()),
        rtol=1e-14,
    )
    # Subtraction
    sub_array = stacked_array1 - stacked_array2
    assert isinstance(sub_array, StackedDistributedArray)
    assert_allclose(
        sub_array.asarray().get(),
        np.subtract(stacked_array1.asarray().get(), stacked_array2.asarray().get()),
        rtol=1e-14,
    )
    # Multiplication
    mult_array = stacked_array1 * stacked_array2
    assert isinstance(mult_array, StackedDistributedArray)
    assert_allclose(
        mult_array.asarray().get(),
        np.multiply(stacked_array1.asarray().get(), stacked_array2.asarray().get()),
        rtol=1e-14,
    )
    # Dot-product
    dot_prod = stacked_array1.dot(stacked_array2)
    assert_allclose(
        dot_prod,
        np.dot(
            stacked_array1.asarray().flatten().get(),
            stacked_array2.asarray().flatten().get(),
        ),
        rtol=1e-14,
    )
    # Norm
    l0norm = stacked_array1.norm(0)
    l1norm = stacked_array1.norm(1)
    l2norm = stacked_array1.norm(2)
    linfnorm = stacked_array1.norm(np.inf)

    assert_allclose(
        l0norm, np.linalg.norm(stacked_array1.asarray().flatten().get(), 0), rtol=1e-14
    )
    assert_allclose(
        l1norm, np.linalg.norm(stacked_array1.asarray().flatten().get(), 1), rtol=1e-14
    )
    assert_allclose(
        l2norm, np.linalg.norm(stacked_array1.asarray().get(), 2), rtol=1e-10
    )
    assert_allclose(
        linfnorm,
        np.linalg.norm(stacked_array1.asarray().flatten().get(), np.inf),
        rtol=1e-14,
    )


def _check_math_out(stacked_array1, stacked_array2):
    """Arithmetic with out= and in-place operators on two stacked arrays"""
    g1 = stacked_array1.asarray().get()
    g2 = stacked_array2.asarray().get()

    for name, op in [
        ("add", np.add),
        ("subtract", np.subtract),
        ("multiply", np.multiply),
    ]:
        # Ops with 2 StackedDistributedArray writing to
        # pre-instantiated StackedDistributedArray
        out = stacked_array1.empty_like()
        res = getattr(stacked_array1, name)(stacked_array2, out=out)
        assert res is out
        assert isinstance(out, StackedDistributedArray)
        assert_allclose(out.asarray().get(), op(g1, g2), rtol=1e-14)

        # Ops with StackedDistributedArray and scalar writing to
        # pre-instantiated StackedDistributedArray
        out = stacked_array1.empty_like()
        getattr(stacked_array1, name)(2.5, out=out)
        assert_allclose(out.asarray().get(), op(g1, 2.5), rtol=1e-14)

        # Check input operands are untouched
        assert_allclose(stacked_array1.asarray().get(), g1, rtol=1e-14)
        assert_allclose(stacked_array2.asarray().get(), g2, rtol=1e-14)

    # Out acting as second operand as well as output
    out = stacked_array2.copy()
    stacked_array1.subtract(out, out=out)
    assert_allclose(out.asarray().get(), g1 - g2, rtol=1e-14)

    # Negation
    assert_allclose((-stacked_array1).asarray().get(), -g1, rtol=1e-14)

    # In-place operators: check correctness and that they keep the storage
    for opname, op in [
        ("__iadd__", np.add),
        ("__isub__", np.subtract),
        ("__imul__", np.multiply),
    ]:
        for other, other_g in [(stacked_array2, g2), (2.5, 2.5)]:
            y = stacked_array1.copy()
            locals_before = [leaf.local_array for leaf in y._leaves()]
            res = getattr(y, opname)(other)
            assert res is y
            assert all(
                leaf.local_array is lb for leaf, lb in zip(y._leaves(), locals_before)
            )
            assert_allclose(y.asarray().get(), op(g1, other_g), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j), (par3), (par3j)])
def test_stacked_math_out_nccl(par):
    """Test arithmetic with out= and in-place operators
    for stacked distributed arrays"""
    distributed_array0 = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array1 = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array0[:] = 0
    distributed_array1[:] = cp.arange(np.prod(distributed_array1.local_shape)).reshape(
        distributed_array1.local_shape
    )

    stacked_array1 = StackedDistributedArray([distributed_array0, distributed_array1])
    stacked_array2 = StackedDistributedArray([distributed_array1, distributed_array0])
    _check_math_out(stacked_array1, stacked_array2)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j), (par3), (par3j)])
def test_stacked_nested_math_out_nccl(par):
    """Test arithmetic with out= and in-place operators
    for nested stacked distributed arrays"""
    distributed_array0 = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array1 = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array2 = DistributedArray(
        global_shape=par["global_shape"],
        base_comm_nccl=nccl_comm,
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array0[:] = 0
    distributed_array1[:] = cp.arange(np.prod(distributed_array1.local_shape)).reshape(
        distributed_array1.local_shape
    )
    distributed_array2[:] = cp.ones(np.prod(distributed_array2.local_shape)).reshape(
        distributed_array1.local_shape
    )

    stacked_array_01 = StackedDistributedArray([distributed_array0, distributed_array1])
    stacked_array_12 = StackedDistributedArray([distributed_array1, distributed_array2])
    stacked_array1 = StackedDistributedArray([stacked_array_01, distributed_array2])
    stacked_array2 = StackedDistributedArray([stacked_array_12, distributed_array0])
    _check_math_out(stacked_array1, stacked_array2)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j), (par3), (par3j)])
@pytest.mark.parametrize("nested", [False, True])
def test_stacked_fused_reductions_nccl(par, nested):
    """Test that dot-product and norm of (nested) stacked distributed arrays
    equal the combination of the per-leaf results"""
    distributed_arrays = []
    for scale in (1, 2, 3):
        distributed_array = DistributedArray(
            global_shape=par["global_shape"],
            base_comm_nccl=nccl_comm,
            partition=par["partition"],
            dtype=par["dtype"],
            axis=par["axis"],
            engine="cupy",
        )
        values = scale * cp.arange(1, np.prod(distributed_array.local_shape) + 1)
        if cp.iscomplexobj(distributed_array.local_array):
            values = values * (1 + 0.5j)
        distributed_array[:] = values.reshape(distributed_array.local_shape)
        distributed_arrays.append(distributed_array)
    distributed_array0, distributed_array1, distributed_array2 = distributed_arrays

    if nested:
        stacked_array1 = StackedDistributedArray(
            [
                distributed_array0,
                StackedDistributedArray([distributed_array1, distributed_array2]),
            ]
        )
        stacked_array2 = StackedDistributedArray(
            [
                distributed_array2,
                StackedDistributedArray([distributed_array0, distributed_array1]),
            ]
        )
    else:
        stacked_array1 = StackedDistributedArray(distributed_arrays)
        stacked_array2 = StackedDistributedArray(distributed_arrays[::-1])

    # Dot-product
    for vdot in (False, True):
        expected = sum(
            a.dot(b, vdot=vdot)
            for a, b in zip(stacked_array1._leaves(), stacked_array2._leaves())
        )
        assert_allclose(
            stacked_array1.dot(stacked_array2, vdot=vdot), expected, rtol=1e-12
        )
    # Norm
    for ord in (None, 2, 1, np.inf):
        norms = np.array([leaf.norm(ord) for leaf in stacked_array1._leaves()])
        if ord in (None, 2):
            expected = np.sqrt(np.sum(norms**2))
        elif ord == 1:
            expected = np.sum(norms)
        else:
            expected = np.max(norms)
        assert_allclose(stacked_array1.norm(ord), expected, rtol=1e-12)
