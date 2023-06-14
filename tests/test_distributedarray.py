"""Test the DistributedArray class
    Designed to run with n processes
    $ mpiexec -n 10 pytest test_distributedarray.py --with-mpi
"""
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_almost_equal

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

par3_1 = {'x': np.random.normal(100, 100, (500, 501)),
          'partition': Partition.SCATTER, 'axis': 1}
par3_2 = {'x': np.random.normal(300, 300, (500, 501)),
          'partition': Partition.SCATTER, 'axis': 1}

par4_1 = {'x': np.random.normal(100, 100, (500, 501)),
          'partition': Partition.BROADCAST, 'axis': 0}
par4_2 = {'x': np.random.normal(300, 300, (500, 501)),
          'partition': Partition.BROADCAST, 'axis': 0}

par5 = {'global_shape': (200, 201, 101),
        'partition': Partition.SCATTER,
        'dtype': np.float64, 'axis': 1}

par5j = {'global_shape': (200, 201, 101),
         'partition': Partition.SCATTER,
         'dtype': np.complex128, 'axis': 2}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2),
                                 (par2j), (par5), (par5j)])
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
    assert (distributed_ones.local_array
            == np.ones(shape=distributed_ones.local_shape,
                       dtype=par['dtype'])).all()
    assert (distributed_ones.asarray()
            == np.ones(shape=par['global_shape'], dtype=par['dtype'])).all()
    # Test for distributed zeroes
    assert (distributed_zeroes.local_array
            == np.zeros(shape=distributed_zeroes.local_shape,
                        dtype=par['dtype'])).all()
    assert (distributed_zeroes.asarray()
            == np.zeros(shape=par['global_shape'], dtype=par['dtype'])).all()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par3_1), (par3_2), (par4_1), (par4_2)])
def test_to_dist(par):
    """Test the ``to_dist`` method"""
    dist_array = DistributedArray.to_dist(x=par['x'],
                                          partition=par['partition'],
                                          axis=par['axis'])
    assert isinstance(dist_array, DistributedArray)
    assert dist_array.global_shape == par['x'].shape
    assert dist_array.axis == par['axis']


@pytest.mark.mpi(minsize=2)
@pytest.mark.parametrize("par1, par2", [(par3_1, par3_2), (par4_1, par4_2)])
def test_distributed_math(par1, par2):
    """Test the Element-Wise Addition, Subtraction and Multiplication"""
    arr1 = DistributedArray.to_dist(x=par1['x'], partition=par1['partition'])
    arr2 = DistributedArray.to_dist(x=par2['x'], partition=par2['partition'])
    # Addition
    sum_array = arr1 + arr2
    # Subtraction
    sub_array = arr1 - arr2
    # Multiplication
    mult_array = arr1 * arr2
    # Global array of Sum with np.add
    assert_array_almost_equal(sum_array.asarray(),
                              np.add(par1['x'], par2['x']), decimal=3)
    # Global array of Subtract with np.subtract
    assert_array_almost_equal(sub_array.asarray(),
                              np.subtract(par1['x'], par2['x']), decimal=3)
    # Global array of Multiplication with np.multiply
    assert_array_almost_equal(mult_array.asarray(),
                              np.multiply(par1['x'], par2['x']), decimal=3)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par1, par2", [(par3_1, par3_2), (par4_1, par4_2)])
def test_distributed_dot(par1, par2):
    """Test Distributed Dot product"""
    pass


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par3_1), (par3_2)])
def test_distributed_norm(par):
    """Test Distributed norm method"""
    arr = DistributedArray.to_dist(x=par['x'], axis=par['axis'])
    assert_array_almost_equal(arr.norm(ord=1), np.linalg.norm(par['x'], ord=1, axis=par['axis']),
                              decimal=3)
    assert_array_almost_equal(arr.norm(ord=np.inf), np.linalg.norm(par['x'], ord=np.inf, axis=par['axis']),
                              decimal=3)
    assert_almost_equal(arr.norm(flatten=True), np.linalg.norm(par['x'].flatten()))
