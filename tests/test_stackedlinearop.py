import numpy as np
from numpy.testing import assert_allclose
from mpi4py import MPI
import pytest

import pylops
import pylops_mpi


np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

par1 = {'ny': 101, 'nx': 101, 'dtype': np.float128}
par1j = {'ny': 101, 'nx': 101, 'dtype': np.complex256}
par2 = {'ny': 301, 'nx': 101, 'dtype': np.float128}
par2j = {'ny': 301, 'nx': 101, 'dtype': np.complex256}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_stackedlinearop(par):
    """Apply various overloaded operators (.H, .T, +, *, conj()) and ensure that the
    returned operator is still of `pylops_mpi.MPIStackedLinearOperator` type
    """
    Op = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = pylops_mpi.MPIBlockDiag(ops=[Op, ])
    FirstDeriv_MPI = pylops_mpi.MPIFirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'])
    SecondDeriv_MPI = pylops_mpi.MPISecondDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'])
    StackedBlockDiag_MPI = pylops_mpi.MPIStackedBlockDiag(ops=[BDiag_MPI, FirstDeriv_MPI, SecondDeriv_MPI])

    assert isinstance(StackedBlockDiag_MPI, pylops_mpi.MPIStackedLinearOperator)
    assert isinstance(StackedBlockDiag_MPI.H, pylops_mpi.MPIStackedLinearOperator)
    assert isinstance(StackedBlockDiag_MPI.T, pylops_mpi.MPIStackedLinearOperator)
    assert isinstance(StackedBlockDiag_MPI * -3, pylops_mpi.MPIStackedLinearOperator)
    assert isinstance(StackedBlockDiag_MPI.conj(), pylops_mpi.MPIStackedLinearOperator)
    assert isinstance(StackedBlockDiag_MPI + StackedBlockDiag_MPI, pylops_mpi.MPIStackedLinearOperator)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_transpose(par):
    """Test the StackedTransposeLinearOperator"""
    Op = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = pylops_mpi.MPIBlockDiag(ops=[Op, ])
    FirstDeriv_MPI = pylops_mpi.MPIFirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'],)
    StackedBlockDiag_MPI = pylops_mpi.MPIStackedBlockDiag(ops=[BDiag_MPI, FirstDeriv_MPI])
    # Tranposed Op
    Top_MPI = StackedBlockDiag_MPI.T

    # For forward mode
    dist_1 = pylops_mpi.DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    dist_1[:] = np.ones(par['ny'])
    dist_2 = pylops_mpi.DistributedArray(global_shape=par['nx'] * par['ny'], dtype=par['dtype'])
    dist_2[:] = np.ones(dist_2.local_shape)
    x = pylops_mpi.StackedDistributedArray(distarrays=[dist_1, dist_2])
    x_global = x.asarray()
    Top_x = Top_MPI @ x
    assert isinstance(Top_x, pylops_mpi.StackedDistributedArray)
    Top_x_np = Top_x.asarray()

    # For adjoint mode
    dist_1 = pylops_mpi.DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    dist_1[:] = np.ones(par['nx'])
    dist_2 = pylops_mpi.DistributedArray(global_shape=par['nx'] * par['ny'], dtype=par['dtype'])
    dist_2[:] = np.ones(dist_2.local_shape)
    y = pylops_mpi.StackedDistributedArray(distarrays=[dist_1, dist_2])
    y_global = y.asarray()
    Top_y = Top_MPI.H @ y
    assert isinstance(Top_y, pylops_mpi.StackedDistributedArray)
    Top_y_np = Top_y.asarray()
    if rank == 0:
        ops = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in
               range(size)]
        BDiag = pylops.BlockDiag(ops=ops)
        FirstDeriv = pylops.FirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'], axis=0)
        StackedBlockDiag = pylops.BlockDiag([BDiag, FirstDeriv])
        Top = StackedBlockDiag.T
        assert_allclose(Top_x_np, Top @ x_global, rtol=1e-14)
        assert_allclose(Top_y_np, Top.H @ y_global, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_scaled(par):
    """Test the StackedScaledLinearOperator"""
    Op = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = pylops_mpi.MPIBlockDiag(ops=[Op, ])
    FirstDeriv_MPI = pylops_mpi.MPIFirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'],)
    StackedBlockDiag_MPI = pylops_mpi.MPIStackedBlockDiag(ops=[BDiag_MPI, FirstDeriv_MPI])
    # Scaled Op
    Sop_MPI = StackedBlockDiag_MPI * -4

    # For forward mode
    dist_1 = pylops_mpi.DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    dist_1[:] = np.ones(par['nx'])
    dist_2 = pylops_mpi.DistributedArray(global_shape=par['nx'] * par['ny'], dtype=par['dtype'])
    dist_2[:] = np.ones(dist_2.local_shape)
    x = pylops_mpi.StackedDistributedArray(distarrays=[dist_1, dist_2])
    x_global = x.asarray()
    Sop_x = Sop_MPI @ x
    assert isinstance(Sop_x, pylops_mpi.StackedDistributedArray)
    Sop_x_np = Sop_x.asarray()

    # For adjoint mode
    dist_1 = pylops_mpi.DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    dist_1[:] = np.ones(par['ny'])
    dist_2 = pylops_mpi.DistributedArray(global_shape=par['nx'] * par['ny'], dtype=par['dtype'])
    dist_2[:] = np.ones(dist_2.local_shape)
    y = pylops_mpi.StackedDistributedArray(distarrays=[dist_1, dist_2])
    y_global = y.asarray()
    Sop_y = Sop_MPI.H @ y
    assert isinstance(Sop_y, pylops_mpi.StackedDistributedArray)
    Sop_y_np = Sop_y.asarray()
    if rank == 0:
        ops = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in
               range(size)]
        BDiag = pylops.BlockDiag(ops=ops)
        FirstDeriv = pylops.FirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'], axis=0)
        StackedBlockDiag = pylops.BlockDiag([BDiag, FirstDeriv])
        Sop = StackedBlockDiag * -4
        assert_allclose(Sop_x_np, Sop @ x_global, rtol=1e-14)
        assert_allclose(Sop_y_np, Sop.H @ y_global, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_conj(par):
    """Test the StackedConjLinearOperator"""
    Op = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = pylops_mpi.MPIBlockDiag(ops=[Op, ])
    FirstDeriv_MPI = pylops_mpi.MPIFirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'],)
    StackedBlockDiag_MPI = pylops_mpi.MPIStackedBlockDiag(ops=[BDiag_MPI, FirstDeriv_MPI])
    # Conj Op
    Cop_MPI = StackedBlockDiag_MPI.conj()

    # For forward mode
    dist_1 = pylops_mpi.DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    dist_1[:] = np.ones(par['nx'])
    dist_2 = pylops_mpi.DistributedArray(global_shape=par['nx'] * par['ny'], dtype=par['dtype'])
    dist_2[:] = np.ones(dist_2.local_shape)
    x = pylops_mpi.StackedDistributedArray(distarrays=[dist_1, dist_2])
    x_global = x.asarray()
    Cop_x = Cop_MPI @ x
    assert isinstance(Cop_x, pylops_mpi.StackedDistributedArray)
    Cop_x_np = Cop_x.asarray()

    # For adjoint mode
    dist_1 = pylops_mpi.DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    dist_1[:] = np.ones(par['ny'])
    dist_2 = pylops_mpi.DistributedArray(global_shape=par['nx'] * par['ny'], dtype=par['dtype'])
    dist_2[:] = np.ones(dist_2.local_shape)
    y = pylops_mpi.StackedDistributedArray(distarrays=[dist_1, dist_2])
    y_global = y.asarray()
    Cop_y = Cop_MPI.H @ y
    assert isinstance(Cop_y, pylops_mpi.StackedDistributedArray)
    Cop_y_np = Cop_y.asarray()
    if rank == 0:
        ops = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in
               range(size)]
        BDiag = pylops.BlockDiag(ops=ops)
        FirstDeriv = pylops.FirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'], axis=0)
        StackedBlockDiag = pylops.BlockDiag([BDiag, FirstDeriv])
        Sop = StackedBlockDiag.conj()
        assert_allclose(Cop_x_np, Sop @ x_global, rtol=1e-14)
        assert_allclose(Cop_y_np, Sop.H @ y_global, rtol=1e-14)
