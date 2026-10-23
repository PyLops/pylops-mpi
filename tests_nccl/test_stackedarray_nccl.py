"""Test the StackedDistributedArray class
    Designed to run with n GPUs (with 1 MPI process per GPU)
    $ mpiexec -n 10 pytest test_stackedarray_nccl.py --with-mpi

This file employs the same test sets as test_stackedarray under NCCL environment
"""
import cupy as cp
import numpy as np
import pytest
from mpi4py import MPI
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
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array1 = DistributedArray(
        global_shape=par["global_shape"],
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array2 = DistributedArray(
        global_shape=[gs * 2 for gs in par["global_shape"]],
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
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array1 = DistributedArray(
        global_shape=par["global_shape"],
        partition=par["partition"],
        dtype=par["dtype"],
        axis=par["axis"],
        engine="cupy",
    )
    distributed_array2 = DistributedArray(
        global_shape=par["global_shape"],
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


def _make_stack(vals, nested=False, partition=Partition.SCATTER, mask=None,
                dtype=np.float64):
    """Stack of arrays (optionally with a nested stack) with given global values"""
    arrs = [DistributedArray.to_dist(x=cp.asarray(np.asarray(v, dtype=dtype)),
                                     base_comm_nccl=nccl_comm,
                                     partition=partition, mask=mask)
            for v in vals]
    if nested:
        return StackedDistributedArray([arrs[0], StackedDistributedArray(arrs[1:])])
    return StackedDistributedArray(arrs)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("nested", [False, True])
def test_stacked_out_nccl(nested, dtype):
    """Stacked arithmetic with out= and in-place operators"""
    size = MPI.COMM_WORLD.Get_size()
    g1 = [np.random.normal(0, 1, (4 * size, 3)) + 1j * np.random.normal(0, 1, (4 * size, 3)),
          np.random.normal(0, 1, (5 * size,)) + 1j * np.random.normal(0, 1, (5 * size,)),
          np.random.normal(0, 1, (3 * size, 2)) + 1j * np.random.normal(0, 1, (3 * size, 2))]
    g2 = [2 * g + 1 for g in g1]
    if dtype == np.float64:
        g1, g2 = [g.real for g in g1], [g.real for g in g2]
    a = _make_stack(g1, nested, dtype=dtype)
    b = _make_stack(g2, nested, dtype=dtype)

    def check(res, expected):
        full = np.hstack([x.ravel() for x in expected])
        assert_allclose(res.asarray().get(), full, rtol=1e-14)

    for name, op in [("add", np.add), ("subtract", np.subtract),
                     ("multiply", np.multiply)]:
        for other, other_g in [(b, g2), (2.5, [2.5] * 3)]:
            out = a.empty_like()
            res = getattr(a, name)(other, out=out)
            assert res is out
            check(out, [op(x, y) for x, y in zip(g1, other_g)])
            # operands are not modified
            check(a, g1)
    check(a + b, [x + y for x, y in zip(g1, g2)])
    check(a - b, [x - y for x, y in zip(g1, g2)])
    check(a * b, [x * y for x, y in zip(g1, g2)])
    check(-a, [-x for x in g1])
    # in-place operators keep the storage of every leaf
    for opname, op in [("__iadd__", np.add), ("__isub__", np.subtract),
                       ("__imul__", np.multiply)]:
        for other, other_g in [(b, g2), (2.5, [2.5] * 3)]:
            y = a.copy()
            locals_before = [leaf.local_array for leaf in y._leaves()]
            res = getattr(y, opname)(other)
            assert res is y
            assert all(leaf.local_array is lb
                       for leaf, lb in zip(y._leaves(), locals_before))
            check(y, [op(x, z) for x, z in zip(g1, other_g)])


def _count_allreduces(fn):
    import pylops_mpi.Distributed as D
    orig = D.nccl_allreduce
    n = [0]

    def counting(*args, **kwargs):
        n[0] += 1
        return orig(*args, **kwargs)
    D.nccl_allreduce = counting
    try:
        res = fn()
    finally:
        D.nccl_allreduce = orig
    return res, n[0]


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("kind", ["scatter", "broadcast", "masked"])
def test_stacked_fused_reductions_nccl(kind, nested, dtype):
    """Fused dot/norm equal the combination of the per-child results,
    both on the fast path (SCATTER) and the fallback path (BROADCAST, masked)"""
    size = MPI.COMM_WORLD.Get_size()
    if kind == "masked" and size % 2 != 0:
        pytest.skip("Number of processes is not divisible by 2")
    partition = Partition.BROADCAST if kind == "broadcast" else Partition.SCATTER
    mask = np.repeat(np.arange(2), size // 2) if kind == "masked" else None
    g1 = [np.random.normal(0, 1, (4 * size, 3)),
          np.random.normal(0, 1, (4 * size,)),
          np.random.normal(0, 1, (6 * size, 2))]
    g2 = [np.random.normal(0, 1, g.shape) for g in g1]
    if dtype == np.complex128:
        g1 = [g + 1j * np.random.normal(0, 1, g.shape) for g in g1]
        g2 = [g + 1j * np.random.normal(0, 1, g.shape) for g in g2]
    if kind == "masked":
        # replicate the data in the two sub-communicators
        g1 = [np.concatenate([g[:g.shape[0] // 2]] * 2) for g in g1]
        g2 = [np.concatenate([g[:g.shape[0] // 2]] * 2) for g in g2]
    a = _make_stack(g1, nested, partition, mask, dtype)
    b = _make_stack(g2, nested, partition, mask, dtype)

    for vdot in (False, True):
        expected = sum(a_.dot(b_, vdot=vdot) for a_, b_ in zip(a._leaves(), b._leaves()))
        res, nred = _count_allreduces(lambda: a.dot(b, vdot=vdot))
        assert_allclose(res, expected, rtol=1e-12)
        if kind == "scatter":
            assert nred == 1
    for ord in (None, 2, 1, np.inf):
        norms = np.array([leaf.norm(ord) for leaf in a._leaves()])
        if ord in (None, 2):
            expected = np.sqrt(np.sum(norms ** 2))
        elif ord == 1:
            expected = np.sum(norms)
        else:
            expected = np.max(norms)
        res, nred = _count_allreduces(lambda: a.norm(ord))
        assert_allclose(res, expected, rtol=1e-12)
        if kind == "scatter" and ord in (None, 2):
            assert nred == 1
