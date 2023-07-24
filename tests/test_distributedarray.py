"""Test the DistributedArray class
    Designed to run with n processes
    $ mpiexec -n 10 pytest test_distributedarray.py --with-mpi
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pylops_mpi import DistributedArray, Partition
from pylops_mpi.DistributedArray import local_split

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

par4 = {'x': np.random.normal(100, 100, (500, 501)),
        'partition': Partition.SCATTER, 'axis': 1}

par4j = {'x': np.random.normal(100, 100, (500, 501)) + 1.0j * np.random.normal(50, 50, (500, 501)),
         'partition': Partition.SCATTER, 'axis': 1}

par5 = {'x': np.random.normal(300, 300, (500, 501)),
        'partition': Partition.SCATTER, 'axis': 1}

par5j = {'x': np.random.normal(300, 300, (500, 501)) + 1.0j * np.random.normal(50, 50, (500, 501)),
         'partition': Partition.SCATTER, 'axis': 1}

par6 = {'x': np.random.normal(100, 100, (500, 500)),
        'partition': Partition.SCATTER, 'axis': 0}

par7 = {'x': np.random.normal(300, 300, (500, 500)),
        'partition': Partition.SCATTER, 'axis': 0}

par8 = {'x': np.random.normal(100, 100, (1000,)),
        'partition': Partition.SCATTER, 'axis': 0}

par9 = {'x': np.random.normal(300, 300, (1000,)),
        'partition': Partition.SCATTER, 'axis': 0}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2),
                                 (par2j), (par3), (par3j)])
def test_creation(par):
    """Test creation of local arrays"""
    distributed_array = DistributedArray(global_shape=par['global_shape'],
                                         partition=par['partition'],
                                         dtype=par['dtype'], axis=par['axis'])
    loc_shape = local_split(distributed_array.global_shape,
                            distributed_array.base_comm,
                            distributed_array.partition,
                            distributed_array.axis)
    assert distributed_array.global_shape == par['global_shape']
    assert distributed_array.local_shape == loc_shape
    assert isinstance(distributed_array, DistributedArray)
    # Distributed array of ones
    distributed_ones = DistributedArray(global_shape=par['global_shape'],
                                        partition=par['partition'],
                                        dtype=par['dtype'], axis=par['axis'])
    distributed_ones[:] = 1
    # Distributed array of zeroes
    distributed_zeroes = DistributedArray(global_shape=par['global_shape'],
                                          partition=par['partition'],
                                          dtype=par['dtype'], axis=par['axis'])
    distributed_zeroes[:] = 0
    # Test for distributed ones
    assert isinstance(distributed_ones, DistributedArray)
    assert_allclose(distributed_ones.local_array, np.ones(shape=distributed_ones.local_shape,
                                                          dtype=par['dtype']), rtol=1e-14)
    assert_allclose(distributed_ones.asarray(), np.ones(shape=distributed_ones.global_shape,
                                                        dtype=par['dtype']), rtol=1e-14)
    # Test for distributed zeroes
    assert isinstance(distributed_zeroes, DistributedArray)
    assert_allclose(distributed_zeroes.local_array, np.zeros(shape=distributed_zeroes.local_shape,
                                                             dtype=par['dtype']), rtol=1e-14)
    assert_allclose(distributed_zeroes.asarray(), np.zeros(shape=distributed_zeroes.global_shape,
                                                           dtype=par['dtype']), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par4), (par4j), (par5), (par5j)])
def test_to_dist(par):
    """Test the ``to_dist`` method"""
    dist_array = DistributedArray.to_dist(x=par['x'],
                                          partition=par['partition'],
                                          axis=par['axis'])
    assert isinstance(dist_array, DistributedArray)
    assert dist_array.global_shape == par['x'].shape
    assert dist_array.axis == par['axis']


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par1, par2", [(par4, par5), (par4j, par5j)])
def test_distributed_math(par1, par2):
    """Test the Element-Wise Addition, Subtraction and Multiplication"""
    arr1 = DistributedArray.to_dist(x=par1['x'], partition=par1['partition'])
    arr2 = DistributedArray.to_dist(x=par2['x'], partition=par2['partition'])
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
    assert_allclose(sum_array.asarray(), np.add(par1['x'], par2['x']),
                    rtol=1e-14)
    # Global array of Subtract with np.subtract
    assert_allclose(sub_array.asarray(), np.subtract(par1['x'], par2['x']),
                    rtol=1e-14)
    # Global array of Multiplication with np.multiply
    assert_allclose(mult_array.asarray(), np.multiply(par1['x'], par2['x']),
                    rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par1, par2", [(par6, par7), (par8, par9)])
def test_distributed_dot(par1, par2):
    """Test Distributed Dot product"""
    arr1 = DistributedArray.to_dist(x=par1['x'], partition=par1['partition'], axis=par1['axis'])
    arr2 = DistributedArray.to_dist(x=par2['x'], partition=par2['partition'], axis=par2['axis'])
    assert_allclose(arr1.dot(arr2), np.dot(par1['x'].flatten(), par2['x'].flatten()), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par4), (par4j), (par5), (par5j),
                                 (par6), (par7), (par8), (par9)])
def test_distributed_norm(par):
    """Test Distributed numpy.linalg.norm method"""
    arr = DistributedArray.to_dist(x=par['x'], axis=par['axis'])
    assert_allclose(arr.norm(ord=1, axis=par['axis']),
                    np.linalg.norm(par['x'], ord=1, axis=par['axis']), rtol=1e-14)
    assert_allclose(arr.norm(ord=np.inf, axis=par['axis']),
                    np.linalg.norm(par['x'], ord=np.inf, axis=par['axis']), rtol=1e-14)
    assert_allclose(arr.norm(), np.linalg.norm(par['x'].flatten()), rtol=1e-14)
