"""Test the MPIBlockDiag and MPIStackedBlockDiag classes
    Designed to run with n GPUs (with 1 MPI process per GPU)
    $ mpiexec -n 10 pytest test_blockdiag_nccl.py --with-mpi

This file employs the same test sets as test_blockdiag under NCCL environment
"""
from mpi4py import MPI
import numpy as np
import cupy as cp
from numpy.testing import assert_allclose
import pytest

import pylops
import pylops_mpi
from pylops_mpi.utils.dottest import dottest
from pylops_mpi.utils._nccl import initialize_nccl_comm

nccl_comm = initialize_nccl_comm()

par1 = {'ny': 101, 'nx': 101, 'dtype': np.float64}
# par1j = {'ny': 101, 'nx': 101, 'dtype': np.complex128}
par2 = {'ny': 301, 'nx': 101, 'dtype': np.float64}
# par2j = {'ny': 301, 'nx': 101, 'dtype': np.complex128}

np.random.seed(42)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_blockdiag_nccl(par):
    """Test the MPIBlockDiag with NCCL"""
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    Op = pylops.MatrixMult(A=((rank + 1) * cp.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = pylops_mpi.MPIBlockDiag(ops=[Op, ], )

    x = pylops_mpi.DistributedArray(global_shape=size * par['nx'],
                                    base_comm_nccl=nccl_comm,
                                    dtype=par['dtype'],
                                    engine="cupy")
    x[:] = cp.ones(shape=par['nx'], dtype=par['dtype'])
    x_global = x.asarray()

    y = pylops_mpi.DistributedArray(global_shape=size * par['ny'],
                                    base_comm_nccl=nccl_comm,
                                    dtype=par['dtype'],
                                    engine="cupy")
    y[:] = cp.ones(shape=par['ny'], dtype=par['dtype'])
    y_global = y.asarray()

    # Forward
    x_mat = BDiag_MPI @ x
    # Adjoint
    y_rmat = BDiag_MPI.H @ y
    assert isinstance(x_mat, pylops_mpi.DistributedArray)
    assert isinstance(y_rmat, pylops_mpi.DistributedArray)
    # Dot test
    dottest(BDiag_MPI, x, y, size * par['ny'], size * par['nx'])

    x_mat_mpi = x_mat.asarray()
    y_rmat_mpi = y_rmat.asarray()

    if rank == 0:
        ops = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in range(size)]
        BDiag = pylops.BlockDiag(ops=ops)

        x_mat_np = BDiag @ x_global.get()
        y_rmat_np = BDiag.H @ y_global.get()
        assert_allclose(x_mat_mpi.get(), x_mat_np, rtol=1e-14)
        assert_allclose(y_rmat_mpi.get(), y_rmat_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_stacked_blockdiag_nccl(par):
    """Tests for MPIStackedBlogDiag with NCCL"""
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    Op = pylops.MatrixMult(A=((rank + 1) * cp.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = pylops_mpi.MPIBlockDiag(ops=[Op, ], )
    FirstDeriv_MPI = pylops_mpi.MPIFirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'])
    StackedBDiag_MPI = pylops_mpi.MPIStackedBlockDiag(ops=[BDiag_MPI, FirstDeriv_MPI])

    dist1 = pylops_mpi.DistributedArray(global_shape=size * par['nx'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    dist1[:] = cp.ones(dist1.local_shape, dtype=par['dtype'])
    dist2 = pylops_mpi.DistributedArray(global_shape=par['nx'] * par['ny'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    dist2[:] = cp.ones(dist2.local_shape, dtype=par['dtype'])
    x = pylops_mpi.StackedDistributedArray(distarrays=[dist1, dist2])
    x_global = x.asarray()

    dist1 = pylops_mpi.DistributedArray(global_shape=size * par['ny'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    dist1[:] = cp.ones(dist1.local_shape, dtype=par['dtype'])
    dist2 = pylops_mpi.DistributedArray(global_shape=par['nx'] * par['ny'], base_comm_nccl=nccl_comm, dtype=par['dtype'], engine="cupy")
    dist2[:] = cp.ones(dist2.local_shape, dtype=par['dtype'])
    y = pylops_mpi.StackedDistributedArray(distarrays=[dist1, dist2])
    y_global = y.asarray()

    # Forward
    x_mat = StackedBDiag_MPI @ x
    # Adjoint
    y_rmat = StackedBDiag_MPI.H @ y
    assert isinstance(x_mat, pylops_mpi.StackedDistributedArray)
    assert isinstance(y_rmat, pylops_mpi.StackedDistributedArray)
    # Dot test
    dottest(StackedBDiag_MPI, x, y, size * par['ny'] + par['nx'] * par['ny'], size * par['nx'] + par['nx'] * par['ny'])

    x_mat_mpi = x_mat.asarray()
    y_rmat_mpi = y_rmat.asarray()

    if rank == 0:
        ops = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in range(size)]
        BDiag = pylops.BlockDiag(ops=ops)
        FirstDeriv = pylops.FirstDerivative(dims=(par['ny'], par['nx']), axis=0, dtype=par['dtype'])
        BDiag_final = pylops.BlockDiag([BDiag, FirstDeriv])
        x_mat_np = BDiag_final @ x_global.get()
        y_rmat_np = BDiag_final.H @ y_global.get()
        assert_allclose(x_mat_mpi.get(), x_mat_np, rtol=1e-14)
        assert_allclose(y_rmat_mpi.get(), y_rmat_np, rtol=1e-14)
