import numpy as np
from numpy.testing import assert_allclose
from mpi4py import MPI
import pytest

import pylops
import pylops_mpi

par1 = {'ny': 101, 'nx': 101, 'imag': 0, 'dtype': np.float64}
par1j = {'ny': 101, 'nx': 101, 'imag': 1j, 'dtype': np.complex128}
par2 = {'ny': 301, 'nx': 101, 'imag': 0, 'dtype': np.float64}
par2j = {'ny': 301, 'nx': 101, 'imag': 1j, 'dtype': np.complex128}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_vstack(par):
    """Test the MPIVStack operator"""
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    A = np.ones(shape=(par['ny'], par['nx'])) + par['imag'] * np.ones(shape=(par['ny'], par['nx']))
    Op = pylops.MatrixMult(A=((rank + 1) * A).astype(par['dtype']))
    VStack_MPI = pylops_mpi.MPIVStack(ops=[Op, ])

    # Broadcasted DistributedArray(global_shape == local_shape)
    x = pylops_mpi.DistributedArray(global_shape=par['nx'],
                                    partition=pylops_mpi.Partition.BROADCAST,
                                    dtype=par['dtype'])
    x[:] = np.ones(shape=par['nx'], dtype=par['dtype'])
    x_global = x.asarray()

    # Scattered DistributedArray
    y = pylops_mpi.DistributedArray(global_shape=size * par['ny'],
                                    partition=pylops_mpi.Partition.SCATTER,
                                    dtype=par['dtype'])
    y[:] = np.ones(shape=par['ny'], dtype=par['dtype'])
    y_global = y.asarray()

    x_mat = VStack_MPI @ x
    y_rmat = VStack_MPI.H @ y
    assert isinstance(x_mat, pylops_mpi.DistributedArray)
    assert isinstance(y_rmat, pylops_mpi.DistributedArray)

    x_mat_mpi = x_mat.asarray()
    y_rmat_mpi = y_rmat.asarray()

    if rank == 0:
        ops = [pylops.MatrixMult(A=((i + 1) * A).astype(par['dtype'])) for i in range(size)]
        VStack = pylops.VStack(ops=ops)
        x_mat_np = VStack @ x_global
        y_rmat_np = VStack.H @ y_global
        assert_allclose(x_mat_mpi, x_mat_np, rtol=1e-14)
        assert_allclose(y_rmat_mpi, y_rmat_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_stacked_vstack(par):
    """Test the MPIStackedVStack operator"""
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    A = np.ones(shape=(par['ny'], par['nx'])) + par['imag'] * np.ones(shape=(par['ny'], par['nx']))
    Op = pylops.MatrixMult(A=((rank + 1) * A).astype(par['dtype']))
    VStack_MPI = pylops_mpi.MPIVStack(ops=[Op, ])
    StackedVStack_MPI = pylops_mpi.MPIStackedVStack([VStack_MPI, VStack_MPI])

    # Broadcasted DistributedArray(global_shape == local_shape)
    x = pylops_mpi.DistributedArray(global_shape=par['nx'],
                                    partition=pylops_mpi.Partition.BROADCAST,
                                    dtype=par['dtype'])
    x[:] = np.ones(shape=par['nx'], dtype=par['dtype'])
    x_global = x.asarray()

    # Stacked DistributedArray
    dist1 = pylops_mpi.DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    dist1[:] = np.ones(dist1.local_shape, dtype=par['dtype'])
    dist2 = pylops_mpi.DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    dist2[:] = np.ones(dist1.local_shape, dtype=par['dtype'])
    y = pylops_mpi.StackedDistributedArray(distarrays=[dist1, dist2])
    y_global = y.asarray()

    x_mat = StackedVStack_MPI @ x
    y_rmat = StackedVStack_MPI.H @ y
    assert isinstance(x_mat, pylops_mpi.StackedDistributedArray)
    assert isinstance(y_rmat, pylops_mpi.DistributedArray)

    x_mat_mpi = x_mat.asarray()
    y_rmat_mpi = y_rmat.asarray()

    if rank == 0:
        ops = [pylops.MatrixMult(A=((i + 1) * A).astype(par['dtype'])) for i in range(size)]
        VStack = pylops.VStack(ops=ops)
        VStack_final = pylops.VStack(ops=[VStack, VStack])
        x_mat_np = VStack_final @ x_global
        y_rmat_np = VStack_final.H @ y_global
        assert_allclose(x_mat_mpi, x_mat_np, rtol=1e-14)
        assert_allclose(y_rmat_mpi, y_rmat_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_hstack(par):
    """Test the MPIHStack operator"""
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    A = np.ones(shape=(par['ny'], par['nx'])) + par['imag'] * np.ones(shape=(par['ny'], par['nx']))
    Op = pylops.MatrixMult(A=((rank + 1) * A).astype(par['dtype']))
    HStack_MPI = pylops_mpi.MPIHStack(ops=[Op, ])

    # Scattered DistributedArray
    x = pylops_mpi.DistributedArray(global_shape=size * par['nx'],
                                    partition=pylops_mpi.Partition.SCATTER,
                                    dtype=par['dtype'])
    x[:] = np.ones(shape=par['nx'], dtype=par['dtype'])
    x_global = x.asarray()

    # Broadcasted DistributedArray(global_shape == local_shape)
    y = pylops_mpi.DistributedArray(global_shape=par['ny'],
                                    partition=pylops_mpi.Partition.BROADCAST,
                                    dtype=par['dtype'])
    y[:] = np.ones(shape=par['ny'], dtype=par['dtype'])
    y_global = y.asarray()

    x_mat = HStack_MPI @ x
    y_rmat = HStack_MPI.H @ y
    assert isinstance(x_mat, pylops_mpi.DistributedArray)
    assert isinstance(y_rmat, pylops_mpi.DistributedArray)

    x_mat_mpi = x_mat.asarray()
    y_rmat_mpi = y_rmat.asarray()

    if rank == 0:
        ops = [pylops.MatrixMult(A=((i + 1) * A).astype(par['dtype'])) for i in range(size)]
        HStack = pylops.HStack(ops=ops)
        x_mat_np = HStack @ x_global
        y_rmat_np = HStack.H @ y_global
        assert_allclose(x_mat_mpi, x_mat_np, rtol=1e-14)
        assert_allclose(y_rmat_mpi, y_rmat_np, rtol=1e-14)
