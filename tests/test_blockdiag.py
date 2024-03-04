from mpi4py import MPI
import numpy as np
from numpy.testing import assert_allclose
np.random.seed(42)

import pytest

import pylops
import pylops_mpi

par1 = {'ny': 101, 'nx': 101, 'dtype': np.float64}
par1j = {'ny': 101, 'nx': 101, 'dtype': np.complex128}
par2 = {'ny': 301, 'nx': 101, 'dtype': np.float64}
par2j = {'ny': 301, 'nx': 101, 'dtype': np.complex128}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_blockdiag(par):
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    Op = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = pylops_mpi.MPIBlockDiag(ops=[Op, ])

    x = pylops_mpi.DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    x[:] = np.ones(shape=par['nx'], dtype=par['dtype'])
    x_global = x.asarray()

    y = pylops_mpi.DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    y[:] = np.ones(shape=par['ny'], dtype=par['dtype'])
    y_global = y.asarray()

    x_mat = BDiag_MPI @ x
    y_rmat = BDiag_MPI.H @ y
    assert isinstance(x_mat, pylops_mpi.DistributedArray)
    assert isinstance(y_rmat, pylops_mpi.DistributedArray)

    x_mat_mpi = x_mat.asarray()
    y_rmat_mpi = y_rmat.asarray()

    if rank == 0:
        ops = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in range(size)]
        BDiag = pylops.BlockDiag(ops=ops)

        x_mat_np = BDiag @ x_global
        y_rmat_np = BDiag.H @ y_global
        assert_allclose(x_mat_mpi, x_mat_np, rtol=1e-14)
        assert_allclose(y_rmat_mpi, y_rmat_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_stacked_blockdiag(par):
    """Tests for MPIStackedBlogDiag"""
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    Op = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = pylops_mpi.MPIBlockDiag(ops=[Op, ])
    FirstDeriv_MPI = pylops_mpi.MPIFirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'])
    StackedBDiag_MPI = pylops_mpi.MPIStackedBlockDiag(ops=[BDiag_MPI, FirstDeriv_MPI])

    dist1 = pylops_mpi.DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    dist1[:] = np.ones(dist1.local_shape, dtype=par['dtype'])
    dist2 = pylops_mpi.DistributedArray(global_shape=par['nx'] * par['ny'], dtype=par['dtype'])
    dist2[:] = np.ones(dist2.local_shape, dtype=par['dtype'])
    x = pylops_mpi.StackedDistributedArray(distarrays=[dist1, dist2])
    x_global = x.asarray()

    dist1 = pylops_mpi.DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    dist1[:] = np.ones(dist1.local_shape, dtype=par['dtype'])
    dist2 = pylops_mpi.DistributedArray(global_shape=par['nx'] * par['ny'], dtype=par['dtype'])
    dist2[:] = np.ones(dist2.local_shape, dtype=par['dtype'])
    y = pylops_mpi.StackedDistributedArray(distarrays=[dist1, dist2])
    y_global = y.asarray()

    x_mat = StackedBDiag_MPI @ x
    y_rmat = StackedBDiag_MPI.H @ y
    assert isinstance(x_mat, pylops_mpi.StackedDistributedArray)
    assert isinstance(y_rmat, pylops_mpi.StackedDistributedArray)

    x_mat_mpi = x_mat.asarray()
    y_rmat_mpi = y_rmat.asarray()

    if rank == 0:
        ops = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in range(size)]
        BDiag = pylops.BlockDiag(ops=ops)
        FirstDeriv = pylops.FirstDerivative(dims=(par['ny'], par['nx']), axis=0, dtype=par['dtype'])
        BDiag_final = pylops.BlockDiag([BDiag, FirstDeriv])
        x_mat_np = BDiag_final @ x_global
        y_rmat_np = BDiag_final.H @ y_global
        assert_allclose(x_mat_mpi, x_mat_np, rtol=1e-14)
        assert_allclose(y_rmat_mpi, y_rmat_np, rtol=1e-14)
