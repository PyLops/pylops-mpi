"""Test the StackedDistributedArray class
    Designed to run with n GPUs (with 1 MPI process per GPU)
    $ mpiexec -n 10 pytest test_stackedarray_nccl.py --with-mpi

This file employs the same test sets as test_stackedarray under NCCL environment
"""
import numpy as np
import cupy as cp
import pytest
from numpy.testing import assert_allclose

from pylops_mpi import DistributedArray, Partition, StackedDistributedArray
from pylops_mpi.utils._nccl import initialize_nccl_comm

nccl_comm = initialize_nccl_comm()

np.random.seed(42)

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
def test_creation_nccl(par):
    """Test creation of stacked distributed arrays"""
    # Create stacked array
    distributed_array0 = DistributedArray(global_shape=par['global_shape'],
                                          base_comm_nccl=nccl_comm,
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine="cupy")
    distributed_array1 = DistributedArray(global_shape=par['global_shape'],
                                          base_comm_nccl=nccl_comm,
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine="cupy")
    distributed_array0[:] = 0
    distributed_array1[:] = 1

    stacked_arrays = StackedDistributedArray([distributed_array0, distributed_array1])
    assert isinstance(stacked_arrays, StackedDistributedArray)
    assert_allclose(stacked_arrays[0].local_array.get(),
                    np.zeros(shape=distributed_array0.local_shape,
                             dtype=par['dtype']), rtol=1e-14)
    assert_allclose(stacked_arrays[1].local_array.get(),
                    np.ones(shape=distributed_array1.local_shape,
                            dtype=par['dtype']), rtol=1e-14)

    # Modify array in place
    distributed_array0[:] = 2
    assert_allclose(stacked_arrays[0].local_array.get(),
                    2 * np.ones(shape=distributed_array0.local_shape,
                                dtype=par['dtype']), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2),
                                 (par2j), (par3), (par3j)])
def test_stacked_math_nccl(par):
    """Test the Element-Wise Addition, Subtraction and Multiplication, Dot-product, Norm"""
    distributed_array0 = DistributedArray(global_shape=par['global_shape'],
                                          base_comm_nccl=nccl_comm,
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine="cupy")
    distributed_array1 = DistributedArray(global_shape=par['global_shape'],
                                          base_comm_nccl=nccl_comm,
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'],
                                          engine="cupy")
    distributed_array0[:] = 0
    distributed_array1[:] = cp.arange(np.prod(distributed_array1.local_shape)).reshape(distributed_array1.local_shape)

    stacked_array1 = StackedDistributedArray([distributed_array0, distributed_array1])
    stacked_array2 = StackedDistributedArray([distributed_array1, distributed_array0])

    # Addition
    sum_array = stacked_array1 + stacked_array2
    assert isinstance(sum_array, StackedDistributedArray)
    assert_allclose(sum_array.asarray().get(), np.add(stacked_array1.asarray().get(),
                    stacked_array2.asarray().get()), rtol=1e-14)
    # Subtraction
    sub_array = stacked_array1 - stacked_array2
    assert isinstance(sub_array, StackedDistributedArray)
    assert_allclose(sub_array.asarray().get(), np.subtract(stacked_array1.asarray().get(),
                    stacked_array2.asarray().get()), rtol=1e-14)
    # Multiplication
    mult_array = stacked_array1 * stacked_array2
    assert isinstance(mult_array, StackedDistributedArray)
    assert_allclose(mult_array.asarray().get(), np.multiply(stacked_array1.asarray().get(),
                    stacked_array2.asarray().get()), rtol=1e-14)
    # Dot-product
    dot_prod = stacked_array1.dot(stacked_array2)
    assert_allclose(dot_prod.get(), np.dot(stacked_array1.asarray().flatten().get(),
                    stacked_array2.asarray().flatten().get()), rtol=1e-14)
    # Norm
    l0norm = stacked_array1.norm(0)
    l1norm = stacked_array1.norm(1)
    l2norm = stacked_array1.norm(2)

    linfnorm = stacked_array1.norm(np.inf)
    assert_allclose(l0norm.get(), np.linalg.norm(stacked_array1.asarray().flatten().get(), 0),
                    rtol=1e-14)
    assert_allclose(l1norm.get(), np.linalg.norm(stacked_array1.asarray().flatten().get(), 1),
                    rtol=1e-14)
    assert_allclose(l2norm.get(), np.linalg.norm(stacked_array1.asarray().get(), 2),
                    rtol=1e-10)  # needed to raise it due to how partial norms are combined (with power applied)
    assert_allclose(linfnorm.get(), np.linalg.norm(stacked_array1.asarray().flatten().get(), np.inf),
                    rtol=1e-14)
