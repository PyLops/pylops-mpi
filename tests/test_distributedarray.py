"""Test the Distributed Array class
    Designed to run with n processes:
    $ mpiexec -n 4 pytest -m mpi test_distributedarray.py
"""
import pytest
import numpy as np
from pylops_mpi.DistributedArray import DistributedArray

par1 = {'global_shape': (1000, 1000), 'partition': 'S', 'dtype': np.float64}
par1j = {'global_shape': (1000, 1000), 'partition': 'S', 'dtype': np.complex128}

par2 = {'global_shape': (1000, 1000), 'partition': 'B', 'dtype': np.float64}
par2j = {'global_shape': (1000, 1000), 'partition': 'B', 'dtype': np.complex128}

par3 = {'x': np.random.normal(100, 100, (1000, 1000)), 'partition': 'S'}
par4 = {'x': np.random.normal(100, 100, (1000, 1000)), 'partition': 'B'}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_creation(par):
    """Test creation of local arrays"""
    distributed_array = DistributedArray(global_shape=par['global_shape'],
                                         partition=par['partition'], dtype=par['dtype'])
    assert distributed_array.local_shape == (100, 1000)
    assert distributed_array.global_shape == (1000, 1000)


@pytest.mark.mpi(minsize=2)
@pytest.mark.parametrize("par", [(par3, par4)])
def test_to_dist(par):
    """Test the ``to_dist`` method"""
    dist_array = DistributedArray.to_dist(x=par['x'], partition=par['partition'])





