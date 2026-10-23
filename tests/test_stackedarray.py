"""Test the StackedDistributedArray class
    Designed to run with n processes
    $ mpiexec -n 10 pytest test_stackedarray.py --with-mpi
"""
import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_allclose

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_allclose

    backend = "numpy"
import numpy as npp
import pytest

from mpi4py import MPI
from pylops_mpi import DistributedArray, Partition, StackedDistributedArray

np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()
if backend == "cupy":
    device_id = rank % np.cuda.runtime.getDeviceCount()
    np.cuda.Device(device_id).use()


par1 = {'global_shape': (500, 501),
        'partition': Partition.SCATTER, 'dtype': np.float64,
        'axis': 1}
par1j = {'global_shape': (501, 500),
         'partition': Partition.SCATTER, 'dtype': np.complex128,
         'axis': 0}
par2 = {'global_shape': (500, 501),
        'partition': Partition.BROADCAST, 'dtype': np.float64,
        'axis': 1}
par2j = {'global_shape': (501, 500),
         'partition': Partition.BROADCAST, 'dtype': np.complex128,
         'axis': 0}

par3 = {'global_shape': (100, 101, 51),
        'partition': Partition.SCATTER,
        'dtype': np.float64, 'axis': 1}

par3j = {'global_shape': (100, 101, 51),
         'partition': Partition.SCATTER,
         'dtype': np.complex128, 'axis': 2}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1),])
def test_wrong_engine(par):
    """Test that an error is raised if a stacked distributed array
    is created from distributed arrays with different engine"""
    # Create stacked array
    distributed_array0 = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)
    distributed_array1 = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)

    # Force engine (done this way as passing it during creation would
    # raise an error if not numpy/cupy and cupy cannot be chosen for 
    # CPU tests)
    distributed_array0._engine = "foo"
    
    with pytest.raises(ValueError, match="Stacked arrays have "):
        _ = StackedDistributedArray([distributed_array0, distributed_array1])


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2),
                                 (par2j), (par3), (par3j)])
def test_creation(par):
    """Test creation of stacked distributed arrays"""
    # Create stacked array
    distributed_array0 = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)
    distributed_array1 = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)
    distributed_array0[:] = 0
    distributed_array1[:] = 1

    stacked_arrays = StackedDistributedArray([distributed_array0, distributed_array1])
    assert isinstance(stacked_arrays, StackedDistributedArray)
    assert stacked_arrays.global_shape == tuple(gs * 2 for gs in par['global_shape'])
    assert_allclose(stacked_arrays[0].local_array,
                    np.zeros(shape=distributed_array0.local_shape,
                             dtype=par['dtype']), rtol=1e-14)
    assert_allclose(stacked_arrays[1].local_array,
                    np.ones(shape=distributed_array1.local_shape,
                            dtype=par['dtype']), rtol=1e-14)

    # Asarray
    stacked_arrays_local = stacked_arrays.asarray()
    stacked_arrays_local_bench = np.hstack([
        np.zeros(shape=distributed_array0.global_shape, dtype=par['dtype']).ravel(),
        np.ones(shape=distributed_array1.global_shape, dtype=par['dtype']).ravel(),
        ])
    assert_allclose(stacked_arrays_local, stacked_arrays_local_bench, rtol=1e-14)

    # Modify array in place
    distributed_array0[:] = 2
    assert_allclose(stacked_arrays[0].local_array,
                    2 * np.ones(shape=distributed_array0.local_shape,
                                dtype=par['dtype']), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2),
                                 (par2j), (par3), (par3j)])
def test_stacked_math(par):
    """Test Element-Wise Addition, Subtraction and Multiplication, Dot-product, Norm
    for stacked distributed arrays"""
    distributed_array0 = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)
    distributed_array1 = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)
    distributed_array0[:] = 0
    distributed_array1[:] = np.arange(npp.prod(distributed_array1.local_shape)).reshape(distributed_array1.local_shape)

    stacked_array1 = StackedDistributedArray([distributed_array0, distributed_array1])
    stacked_array2 = StackedDistributedArray([distributed_array1, distributed_array0])

    # Addition
    sum_array = stacked_array1 + stacked_array2
    assert isinstance(sum_array, StackedDistributedArray)
    assert_allclose(sum_array.asarray(), np.add(stacked_array1.asarray(),
                                                stacked_array2.asarray()),
                    rtol=1e-14)
    # Subtraction
    sub_array = stacked_array1 - stacked_array2
    assert isinstance(sub_array, StackedDistributedArray)
    assert_allclose(sub_array.asarray(), np.subtract(stacked_array1.asarray(),
                                                     stacked_array2.asarray()),
                    rtol=1e-14)
    # Multiplication
    mult_array = stacked_array1 * stacked_array2
    assert isinstance(mult_array, StackedDistributedArray)
    assert_allclose(mult_array.asarray(), np.multiply(stacked_array1.asarray(),
                                                      stacked_array2.asarray()),
                    rtol=1e-14)
    # Dot-product
    dot_prod = stacked_array1.dot(stacked_array2)
    assert_allclose(dot_prod, np.dot(stacked_array1.asarray().flatten(),
                                     stacked_array2.asarray().flatten()),
                    rtol=1e-14)
    # Norm
    l0norm = stacked_array1.norm(0)
    l1norm = stacked_array1.norm(1)
    l2norm = stacked_array1.norm(2)

    # TODO (tharitt): FAIL with CuPy + MPI for inf norm - see test_distributedarray.py
    # test_distributed_nrom(par) as well
#     linfnorm = stacked_array1.norm(np.inf)
    assert_allclose(l0norm, np.linalg.norm(stacked_array1.asarray().flatten(), 0),
                    rtol=1e-14)
    assert_allclose(l1norm, np.linalg.norm(stacked_array1.asarray().flatten(), 1),
                    rtol=1e-14)
    assert_allclose(l2norm, np.linalg.norm(stacked_array1.asarray(), 2),
                    rtol=1e-10)  # needed to raise it due to how partial norms are combined (with power applied)
    # TODO (tharitt): FAIL at inf norm - see above
#     assert_allclose(linfnorm, np.linalg.norm(stacked_array1.asarray().flatten(), np.inf),
#                     rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1),])
def test_wrong_engine_nested(par):
    """Test that an error is raised if a nested stacked distributed array
    is created from distributed arrays with different engine"""
    # Create stacked array
    distributed_array0 = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)
    distributed_array1 = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)
    distributed_array2 = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)

    # Force engine (done this way as passing it during creation would
    # raise an error if not numpy/cupy and cupy cannot be chosen for 
    # CPU tests)
    distributed_array2._engine = "foo"
    stacked_arrays_01 = StackedDistributedArray([distributed_array0, distributed_array1])
    
    with pytest.raises(ValueError, match="Stacked arrays have "):
        _ = StackedDistributedArray([stacked_arrays_01, distributed_array2])
    

@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2),
                                 (par2j), (par3), (par3j)])
def test_creation_nested(par):
    """Test creation of nested stacked distributed arrays"""
    # Create stacked array
    distributed_array0 = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)
    distributed_array1 = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)
    distributed_array2 = DistributedArray(global_shape=[gs * 2 for gs in par['global_shape']],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)
    distributed_array0[:] = 0
    distributed_array1[:] = 1
    distributed_array2[:] = 2

    stacked_arrays_01 = StackedDistributedArray([distributed_array0, distributed_array1])
    stacked_arrays = StackedDistributedArray([stacked_arrays_01, distributed_array2])
    assert isinstance(stacked_arrays, StackedDistributedArray)
    assert stacked_arrays.global_shape == tuple(gs * 4 for gs in par['global_shape'])
    assert_allclose(stacked_arrays[0][0].local_array,
                    np.zeros(shape=distributed_array0.local_shape,
                             dtype=par['dtype']), rtol=1e-14)
    assert_allclose(stacked_arrays[0][1].local_array,
                    np.ones(shape=distributed_array1.local_shape,
                             dtype=par['dtype']), rtol=1e-14)
    assert_allclose(stacked_arrays[1].local_array,
                    2 * np.ones(shape=distributed_array2.local_shape,
                                dtype=par['dtype']), rtol=1e-14)

    # Asarray
    stacked_arrays_local = stacked_arrays.asarray()
    stacked_arrays_local_bench = np.hstack([
        np.zeros(shape=distributed_array0.global_shape, dtype=par['dtype']).ravel(),
        np.ones(shape=distributed_array1.global_shape, dtype=par['dtype']).ravel(),
        2 * np.ones(shape=distributed_array2.global_shape, dtype=par['dtype']).ravel(),
        ])
    assert_allclose(stacked_arrays_local, stacked_arrays_local_bench, rtol=1e-14)
    
    # Modify array in place
    distributed_array0[:] = 2
    assert_allclose(stacked_arrays[0][0].local_array,
                    2 * np.ones(shape=distributed_array0.local_shape,
                                dtype=par['dtype']), rtol=1e-14)
    distributed_array2[:] = 4
    assert_allclose(stacked_arrays[1].local_array,
                    4 * np.ones(shape=distributed_array2.local_shape,
                                dtype=par['dtype']), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2),
                                 (par2j), (par3), (par3j)])
def test_stacked_nested_math(par):
    """Test Element-Wise Addition, Subtraction and Multiplication, Dot-product, Norm
    for nested stacked distributed arrays"""
    distributed_array0 = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)
    distributed_array1 = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)
    distributed_array2 = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine=backend)
    distributed_array0[:] = 0
    distributed_array1[:] = np.arange(npp.prod(distributed_array1.local_shape)).reshape(distributed_array1.local_shape)
    distributed_array2[:] = np.ones(npp.prod(distributed_array2.local_shape)).reshape(distributed_array1.local_shape)

    stacked_array_01 = StackedDistributedArray([distributed_array0, distributed_array1])
    stacked_array_12 = StackedDistributedArray([distributed_array1, distributed_array2])
    stacked_array1 = StackedDistributedArray([stacked_array_01, distributed_array2])
    stacked_array2 = StackedDistributedArray([stacked_array_12, distributed_array0])

    # Addition
    sum_array = stacked_array1 + stacked_array2
    assert isinstance(sum_array, StackedDistributedArray)
    assert_allclose(sum_array.asarray(), np.add(stacked_array1.asarray(),
                                                stacked_array2.asarray()),
                    rtol=1e-14)
    # Subtraction
    sub_array = stacked_array1 - stacked_array2
    assert isinstance(sub_array, StackedDistributedArray)
    assert_allclose(sub_array.asarray(), np.subtract(stacked_array1.asarray(),
                                                     stacked_array2.asarray()),
                    rtol=1e-14)
    # Multiplication
    mult_array = stacked_array1 * stacked_array2
    assert isinstance(mult_array, StackedDistributedArray)
    assert_allclose(mult_array.asarray(), np.multiply(stacked_array1.asarray(),
                                                      stacked_array2.asarray()),
                    rtol=1e-14)
    # Dot-product
    dot_prod = stacked_array1.dot(stacked_array2)
    assert_allclose(dot_prod, np.dot(stacked_array1.asarray().flatten(),
                                     stacked_array2.asarray().flatten()),
                    rtol=1e-14)
    # Norm
    l0norm = stacked_array1.norm(0)
    l1norm = stacked_array1.norm(1)
    l2norm = stacked_array1.norm(2)
    linfnorm = stacked_array1.norm(np.inf)

    assert_allclose(l0norm, np.linalg.norm(stacked_array1.asarray().flatten(), 0),
                    rtol=1e-14)
    assert_allclose(l1norm, np.linalg.norm(stacked_array1.asarray().flatten(), 1),
                    rtol=1e-14)
    assert_allclose(l2norm, np.linalg.norm(stacked_array1.asarray(), 2),
                    rtol=1e-10)  # needed to raise it due to how partial norms are combined (with power applied)
    assert_allclose(linfnorm, np.linalg.norm(stacked_array1.asarray().flatten(), np.inf),
                    rtol=1e-14)


# Random data for the tests below is generated with numpy (with the same seed in all
# ranks) and then moved to the backend: cupy random generators are per device, so
# the ranks would otherwise generate different "global" arrays
_rng = npp.random.default_rng(42)


def _randn(*shape, cplx=False):
    x = _rng.standard_normal(shape)
    if cplx:
        x = x + 1j * _rng.standard_normal(shape)
    return np.asarray(x)


def _make_stack(vals, nested=False, partition=Partition.SCATTER, mask=None,
                dtype=np.float64):
    """Stack of arrays (optionally with a nested stack) with given global values"""
    arrs = [DistributedArray.to_dist(x=np.asarray(v, dtype=dtype),
                                     partition=partition, mask=mask)
            for v in vals]
    if nested:
        return StackedDistributedArray([arrs[0], StackedDistributedArray(arrs[1:])])
    return StackedDistributedArray(arrs)


def _stack_globals(stack):
    return [np.asarray(a.asarray()) for a in stack.distarrays[:1]] + \
        [np.asarray(s.asarray()) for s in stack.distarrays[1:]]


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("nested", [False, True])
def test_stacked_out(nested, dtype):
    """Stacked arithmetic with out= and in-place operators"""
    size = MPI.COMM_WORLD.Get_size()
    g1 = [_randn(4 * size, 3, cplx=True),
          _randn(5 * size, cplx=True),
          _randn(3 * size, 2, cplx=True)]
    g2 = [2 * g + 1 for g in g1]
    if dtype == np.float64:
        g1, g2 = [g.real for g in g1], [g.real for g in g2]
    a = _make_stack(g1, nested, dtype=dtype)
    b = _make_stack(g2, nested, dtype=dtype)

    def check(res, expected):
        full = np.hstack([x.ravel() for x in expected])
        assert_allclose(res.asarray(), full, rtol=1e-14)

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
    orig = D.mpi_allreduce
    n = [0]

    def counting(*args, **kwargs):
        n[0] += 1
        return orig(*args, **kwargs)
    D.mpi_allreduce = counting
    try:
        res = fn()
    finally:
        D.mpi_allreduce = orig
    return res, n[0]


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("kind", ["scatter", "broadcast", "masked"])
def test_stacked_fused_reductions(kind, nested, dtype):
    """Fused dot/norm equal the combination of the per-child results,
    both on the fast path (SCATTER) and the fallback path (BROADCAST, masked)"""
    size = MPI.COMM_WORLD.Get_size()
    if kind == "masked" and size % 2 != 0:
        pytest.skip("Number of processes is not divisible by 2")
    partition = Partition.BROADCAST if kind == "broadcast" else Partition.SCATTER
    mask = np.repeat(np.arange(2), size // 2) if kind == "masked" else None
    cplx = dtype == np.complex128
    g1 = [_randn(4 * size, 3, cplx=cplx),
          _randn(4 * size, cplx=cplx),
          _randn(6 * size, 2, cplx=cplx)]
    g2 = [_randn(*g.shape, cplx=cplx) for g in g1]
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
    for ord in (None, 2, 1, npp.inf):
        norms = npp.array([leaf.norm(ord) for leaf in a._leaves()])
        if ord in (None, 2):
            expected = npp.sqrt(npp.sum(norms ** 2))
        elif ord == 1:
            expected = npp.sum(norms)
        else:
            expected = npp.max(norms)
        res, nred = _count_allreduces(lambda: a.norm(ord))
        assert_allclose(res, expected, rtol=1e-12)
        if kind == "scatter" and ord in (None, 2):
            assert nred == 1
