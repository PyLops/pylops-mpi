"""Test the DistributedArray class
    Designed to run with n processes
    $ mpiexec -n 10 pytest test_distributedarray.py --with-mpi
"""
import numpy as np
from mpi4py import MPI
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

par6 = {'x': np.random.normal(100, 100, (600, 600)),
        'partition': Partition.SCATTER, 'axis': 0}

par6b = {'x': np.random.normal(100, 100, (600, 600)),
         'partition': Partition.BROADCAST, 'axis': 0}

par7 = {'x': np.random.normal(300, 300, (600, 600)),
        'partition': Partition.SCATTER, 'axis': 0}

par7b = {'x': np.random.normal(300, 300, (600, 600)),
         'partition': Partition.BROADCAST, 'axis': 0}

par8 = {'x': np.random.normal(100, 100, (1200,)),
        'partition': Partition.SCATTER, 'axis': 0}

par8b = {'x': np.random.normal(100, 100, (1200,)),
         'partition': Partition.BROADCAST, 'axis': 0}

par9 = {'x': np.random.normal(300, 300, (1200,)),
        'partition': Partition.SCATTER, 'axis': 0}

par9b = {'x': np.random.normal(300, 300, (1200,)),
         'partition': Partition.BROADCAST, 'axis': 0}


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
@pytest.mark.parametrize("par", [(par1), (par1j), (par2),
                                 (par2j), (par3), (par3j)])
def test_local_shapes(par):
    """Test the `local_shapes` parameter in DistributedArray"""
    # Reverse the local_shapes to test the local_shapes parameter
    loc_shapes = MPI.COMM_WORLD.allgather(local_split(par['global_shape'],
                                                      MPI.COMM_WORLD, par['partition'], par['axis']))[::-1]
    distributed_array = DistributedArray(global_shape=par['global_shape'],
                                         partition=par['partition'],
                                         axis=par['axis'], local_shapes=loc_shapes,
                                         dtype=par['dtype'])
    assert isinstance(distributed_array, DistributedArray)
    assert distributed_array.local_shape == loc_shapes[distributed_array.rank]

    # Distributed ones
    distributed_array[:] = 1
    assert_allclose(distributed_array.local_array, np.ones(loc_shapes[distributed_array.rank],
                                                           dtype=par['dtype']), rtol=1e-14)
    assert_allclose(distributed_array.asarray(), np.ones(par['global_shape'],
                                                         dtype=par['dtype']), rtol=1e-14)


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
@pytest.mark.parametrize("par1, par2", [(par6, par7), (par6b, par7b),
                                        (par8, par9), (par8b, par9b)])
def test_distributed_dot(par1, par2):
    """Test Distributed Dot product"""
    arr1 = DistributedArray.to_dist(x=par1['x'], partition=par1['partition'], axis=par1['axis'])
    arr2 = DistributedArray.to_dist(x=par2['x'], partition=par2['partition'], axis=par2['axis'])
    assert_allclose(arr1.dot(arr2), np.dot(par1['x'].flatten(), par2['x'].flatten()), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par4), (par4j), (par5), (par5j),
                                 (par6), (par6b), (par7), (par7b),
                                 (par8), (par8b), (par9), (par9b)])
def test_distributed_norm(par):
    """Test Distributed numpy.linalg.norm method"""
    arr = DistributedArray.to_dist(x=par['x'], axis=par['axis'])
    assert_allclose(arr.norm(ord=1, axis=par['axis']),
                    np.linalg.norm(par['x'], ord=1, axis=par['axis']), rtol=1e-14)
    assert_allclose(arr.norm(ord=np.inf, axis=par['axis']),
                    np.linalg.norm(par['x'], ord=np.inf, axis=par['axis']), rtol=1e-14)
    assert_allclose(arr.norm(), np.linalg.norm(par['x'].flatten()), rtol=1e-13)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par6), (par8)])
def test_distributed_masked(par):
    """Test Asarray with masked array"""
    # Number of subcommunicators
    size = MPI.COMM_WORLD.Get_size()
    
    # Exclude not handled cases
    shape_axis = par['x'].shape[par['axis']]
    print('shape_axis, size', shape_axis, size, shape_axis % size != 0)
    if shape_axis % size != 0:
        pytest.skip(f"Array dimension to distributed ({shape_axis}) is not  "
                    f"divisible by the number of processes ({size})...")
    if size % 2 == 0:
        nsub = 2
    elif size % 3 == 0:
        nsub = 3
    else:
        pytest.skip(f"Number of processes ({size}) is not divisible "
                    "by 2 or 3...")
    subsize = max(1, MPI.COMM_WORLD.Get_size() // nsub)
    mask = np.repeat(np.arange(nsub), subsize)

    # Replicate x as required in masked arrays
    x = par['x']
    if par['axis'] != 0:
        x = np.swapaxes(x, par['axis'], 0)
    for isub in range(1, nsub):
        x[(x.shape[0] // nsub) * isub:(x.shape[0] // nsub) * (isub + 1)] = x[:x.shape[0] // nsub]
    if par['axis'] != 0:
        x = np.swapaxes(x, 0, par['axis'])

    arr = DistributedArray.to_dist(x=x, partition=par['partition'], mask=mask, axis=par['axis'])

    # Global view
    xloc = arr.asarray()
    assert xloc.shape == x.shape

    # Global masked view
    xmaskedloc = arr.asarray(masked=True)
    xmasked_shape = list(x.shape)
    xmasked_shape[par['axis']] = int(xmasked_shape[par['axis']] // nsub)
    assert xmaskedloc.shape == tuple(xmasked_shape)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par1, par2", [(par6, par7), (par6b, par7b),
                                        (par8, par9), (par8b, par9b)])
def test_distributed_maskeddot(par1, par2):
    """Test Distributed Dot product with masked array"""
    # Number of subcommunicators
    size = MPI.COMM_WORLD.Get_size()
    
    # Exclude not handled cases
    shape_axis = par1['x'].shape[par1['axis']]
    print('shape_axis, size', shape_axis, size, shape_axis % size != 0)
    if shape_axis % size != 0:
        pytest.skip(f"Array dimension to distributed ({shape_axis}) is not  "
                    f"divisible by the number of processes ({size})...")
    if size % 2 == 0:
        nsub = 2
    elif size % 3 == 0:
        nsub = 3
    else:
        pytest.skip(f"Number of processes ({size}) is not divisible "
                    "by 2 or 3...")
    subsize = max(1, MPI.COMM_WORLD.Get_size() // nsub)
    mask = np.repeat(np.arange(nsub), subsize)

    # Replicate x1 and x2 as required in masked arrays
    x1, x2 = par1['x'], par2['x']
    if par1['axis'] != 0:
        x1 = np.swapaxes(x1, par1['axis'], 0)
    for isub in range(1, nsub):
        x1[(x1.shape[0] // nsub) * isub:(x1.shape[0] // nsub) * (isub + 1)] = x1[:x1.shape[0] // nsub]
    if par1['axis'] != 0:
        x1 = np.swapaxes(x1, 0, par1['axis'])
    if par2['axis'] != 0:
        x2 = np.swapaxes(x2, par2['axis'], 0)
    for isub in range(1, nsub):
        x2[(x2.shape[0] // nsub) * isub:(x2.shape[0] // nsub) * (isub + 1)] = x2[:x2.shape[0] // nsub]
    if par2['axis'] != 0:
        x2 = np.swapaxes(x2, 0, par2['axis'])

    arr1 = DistributedArray.to_dist(x=x1, partition=par1['partition'], mask=mask, axis=par1['axis'])
    arr2 = DistributedArray.to_dist(x=x2, partition=par2['partition'], mask=mask, axis=par2['axis'])
    assert_allclose(arr1.dot(arr2), np.dot(x1.flatten(), x2.flatten()) / nsub, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par6), (par6b), (par7), (par7b),
                                 (par8), (par8b), (par9), (par9b)])
def test_distributed_maskednorm(par):
    """Test Distributed numpy.linalg.norm method with masked array"""
    # Number of subcommunicators
    size = MPI.COMM_WORLD.Get_size()

    # Exclude not handled cases
    shape_axis = par['x'].shape[par['axis']]
    print('shape_axis, size', shape_axis, size, shape_axis % size != 0)
    if shape_axis % size != 0:
        pytest.skip(f"Array dimension to distributed ({shape_axis}) is not  "
                    f"divisible by the number of processes ({size})...")
    if size % 2 == 0:
        nsub = 2
    elif size % 3 == 0:
        nsub = 3
    else:
        pytest.skip(f"Number of processes ({size}) is not divisible "
                    "by 2 or 3...")
    subsize = max(1, MPI.COMM_WORLD.Get_size() // nsub)
    mask = np.repeat(np.arange(nsub), subsize)
    # Replicate x as required in masked arrays
    x = par['x']
    if par['axis'] != 0:
        x = np.swapaxes(x, par['axis'], 0)
    for isub in range(1, nsub):
        x[(x.shape[0] // nsub) * isub:(x.shape[0] // nsub) * (isub + 1)] = x[:x.shape[0] // nsub]
    if par['axis'] != 0:
        x = np.swapaxes(x, 0, par['axis'])
    arr = DistributedArray.to_dist(x=x, mask=mask, axis=par['axis'])
    assert_allclose(arr.norm(ord=1, axis=par['axis']),
                    np.linalg.norm(par['x'], ord=1, axis=par['axis']) / nsub, rtol=1e-14)
    assert_allclose(arr.norm(ord=np.inf, axis=par['axis']),
                    np.linalg.norm(par['x'], ord=np.inf, axis=par['axis']), rtol=1e-14)
    assert_allclose(arr.norm(ord=2, axis=par['axis']),
                    np.linalg.norm(par['x'], ord=2, axis=par['axis']) / np.sqrt(nsub), rtol=1e-13)
