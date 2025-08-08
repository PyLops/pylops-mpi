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

par3 = {'global_shape': (200, 201, 101),
        'partition': Partition.SCATTER,
        'dtype': np.float64, 'axis': 1}

par3j = {'global_shape': (200, 201, 101),
         'partition': Partition.SCATTER,
         'dtype': np.complex128, 'axis': 2}


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
    assert_allclose(stacked_arrays[0].local_array,
                    np.zeros(shape=distributed_array0.local_shape,
                             dtype=par['dtype']), rtol=1e-14)
    assert_allclose(stacked_arrays[1].local_array,
                    np.ones(shape=distributed_array1.local_shape,
                            dtype=par['dtype']), rtol=1e-14)

    # Modify array in place
    distributed_array0[:] = 2
    assert_allclose(stacked_arrays[0].local_array,
                    2 * np.ones(shape=distributed_array0.local_shape,
                                dtype=par['dtype']), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2),
                                 (par2j), (par3), (par3j)])
def test_stacked_math(par):
    """Test the Element-Wise Addition, Subtraction and Multiplication, Dot-product, Norm"""
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
