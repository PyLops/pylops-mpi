"""Test the stacking classes
    Designed to run with n GPUs (with 1 MPI process per GPU)
    $ mpiexec -n 10 pytest test_stack_nccl.py --with-mpi

This file employs the same test sets as test_stack under NCCL environment
"""
import numpy as np
import cupy as cp
from numpy.testing import assert_allclose
from mpi4py import MPI
import pytest

import pylops
import pylops_mpi
from pylops_mpi.utils.dottest import dottest
from pylops_mpi.utils._nccl import initialize_nccl_comm

nccl_comm = initialize_nccl_comm()

# imag part is left to future complex-number support
par1 = {'ny': 101, 'nx': 101, 'imag': 0, 'dtype': np.float64}
par2 = {'ny': 301, 'nx': 101, 'imag': 0, 'dtype': np.float64}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_vstack_nccl(par):
    """Test the MPIVStack operator with NCCL"""
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    A_gpu = cp.ones(shape=(par['ny'], par['nx'])) + par['imag'] * cp.ones(shape=(par['ny'], par['nx']))
    Op = pylops.MatrixMult(A=((rank + 1) * A_gpu).astype(par['dtype']))
    VStack_MPI = pylops_mpi.MPIVStack(ops=[Op, ], base_comm_nccl=nccl_comm)

    # Broadcasted DistributedArray(global_shape == local_shape)
    x = pylops_mpi.DistributedArray(global_shape=par['nx'],
                                    base_comm_nccl=nccl_comm,
                                    partition=pylops_mpi.Partition.BROADCAST,
                                    dtype=par['dtype'],
                                    engine="cupy")
    x[:] = cp.ones(shape=par['nx'], dtype=par['dtype'])
    x_global = x.asarray()

    # Scattered DistributedArray
    y = pylops_mpi.DistributedArray(global_shape=size * par['ny'],
                                    base_comm_nccl=nccl_comm,
                                    partition=pylops_mpi.Partition.SCATTER,
                                    dtype=par['dtype'],
                                    engine="cupy")
    y[:] = cp.ones(shape=par['ny'], dtype=par['dtype'])
    y_global = y.asarray()

    # Forward
    x_mat = VStack_MPI @ x
    # Adjoint
    y_rmat = VStack_MPI.H @ y
    assert isinstance(x_mat, pylops_mpi.DistributedArray)
    assert isinstance(y_rmat, pylops_mpi.DistributedArray)
    # Dot test
    dottest(VStack_MPI, x, y, size * par['ny'], par['nx'])

    x_mat_mpi = x_mat.asarray()
    y_rmat_mpi = y_rmat.asarray()

    if rank == 0:
        A = A_gpu.get()
        ops = [pylops.MatrixMult(A=((i + 1) * A).astype(par['dtype'])) for i in range(size)]
        VStack = pylops.VStack(ops=ops)
        x_mat_np = VStack @ x_global.get()
        y_rmat_np = VStack.H @ y_global.get()
        assert_allclose(x_mat_mpi.get(), x_mat_np, rtol=1e-14)
        assert_allclose(y_rmat_mpi.get(), y_rmat_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_stacked_vstack_nccl(par):
    """Test the MPIStackedVStack operator with NCCL"""
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    A_gpu = cp.ones(shape=(par['ny'], par['nx'])) + par['imag'] * cp.ones(shape=(par['ny'], par['nx']))
    Op = pylops.MatrixMult(A=((rank + 1) * A_gpu).astype(par['dtype']))
    VStack_MPI = pylops_mpi.MPIVStack(ops=[Op, ], base_comm_nccl=nccl_comm)
    StackedVStack_MPI = pylops_mpi.MPIStackedVStack([VStack_MPI, VStack_MPI])

    # Broadcasted DistributedArray(global_shape == local_shape)
    x = pylops_mpi.DistributedArray(global_shape=par['nx'],
                                    base_comm_nccl=nccl_comm,
                                    partition=pylops_mpi.Partition.BROADCAST,
                                    dtype=par['dtype'],
                                    engine="cupy")
    x[:] = cp.ones(shape=par['nx'], dtype=par['dtype'])
    x_global = x.asarray()

    # Stacked DistributedArray
    dist1 = pylops_mpi.DistributedArray(global_shape=size * par['ny'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    dist1[:] = cp.ones(dist1.local_shape, dtype=par['dtype'])
    dist2 = pylops_mpi.DistributedArray(global_shape=size * par['ny'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    dist2[:] = cp.ones(dist1.local_shape, dtype=par['dtype'])
    y = pylops_mpi.StackedDistributedArray(distarrays=[dist1, dist2])
    y_global = y.asarray()

    x_mat = StackedVStack_MPI @ x
    y_rmat = StackedVStack_MPI.H @ y
    assert isinstance(x_mat, pylops_mpi.StackedDistributedArray)
    assert isinstance(y_rmat, pylops_mpi.DistributedArray)

    x_mat_mpi = x_mat.asarray()
    y_rmat_mpi = y_rmat.asarray()

    if rank == 0:
        A = A_gpu.get()
        ops = [pylops.MatrixMult(A=((i + 1) * A).astype(par['dtype'])) for i in range(size)]
        VStack = pylops.VStack(ops=ops)
        VStack_final = pylops.VStack(ops=[VStack, VStack])
        x_mat_np = VStack_final @ x_global.get()
        y_rmat_np = VStack_final.H @ y_global.get()
        assert_allclose(x_mat_mpi.get(), x_mat_np, rtol=1e-14)
        assert_allclose(y_rmat_mpi.get(), y_rmat_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_hstack(par):
    """Test the MPIHStack operator with NCCL"""
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    A_gpu = cp.ones(shape=(par['ny'], par['nx'])) + par['imag'] * cp.ones(shape=(par['ny'], par['nx']))
    Op = pylops.MatrixMult(A=((rank + 1) * A_gpu).astype(par['dtype']))
    HStack_MPI = pylops_mpi.MPIHStack(ops=[Op, ], base_comm_nccl=nccl_comm)

    # Scattered DistributedArray
    x = pylops_mpi.DistributedArray(global_shape=size * par['nx'],
                                    base_comm_nccl=nccl_comm,
                                    partition=pylops_mpi.Partition.SCATTER,
                                    dtype=par['dtype'],
                                    engine="cupy")
    x[:] = cp.ones(shape=par['nx'], dtype=par['dtype'])
    x_global = x.asarray()

    # Broadcasted DistributedArray(global_shape == local_shape)
    y = pylops_mpi.DistributedArray(global_shape=par['ny'],
                                    base_comm_nccl=nccl_comm,
                                    partition=pylops_mpi.Partition.BROADCAST,
                                    dtype=par['dtype'],
                                    engine="cupy")
    y[:] = cp.ones(shape=par['ny'], dtype=par['dtype'])
    y_global = y.asarray()

    x_mat = HStack_MPI @ x
    y_rmat = HStack_MPI.H @ y
    assert isinstance(x_mat, pylops_mpi.DistributedArray)
    assert isinstance(y_rmat, pylops_mpi.DistributedArray)

    x_mat_mpi = x_mat.asarray()
    y_rmat_mpi = y_rmat.asarray()

    if rank == 0:
        ops = [pylops.MatrixMult(A=((i + 1) * A_gpu.get()).astype(par['dtype'])) for i in range(size)]
        HStack = pylops.HStack(ops=ops)
        x_mat_np = HStack @ x_global.get()
        y_rmat_np = HStack.H @ y_global.get()
        assert_allclose(x_mat_mpi.get(), x_mat_np, rtol=1e-14)
        assert_allclose(y_rmat_mpi.get(), y_rmat_np, rtol=1e-14)
