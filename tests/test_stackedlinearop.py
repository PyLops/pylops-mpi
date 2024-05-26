import numpy as np
from numpy.testing import assert_allclose
from mpi4py import MPI
import pytest

import pylops
import pylops_mpi


np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

par1 = {'ny': 101, 'nx': 101, 'dtype': np.float64}
par1j = {'ny': 101, 'nx': 101, 'dtype': np.complex128}
par2 = {'ny': 301, 'nx': 101, 'dtype': np.float64}
par2j = {'ny': 301, 'nx': 101, 'dtype': np.complex128}


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
    FirstDeriv_MPI = pylops_mpi.MPIFirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'], )
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
    FirstDeriv_MPI = pylops_mpi.MPIFirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'], )
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
    FirstDeriv_MPI = pylops_mpi.MPIFirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'], )
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


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_power(par):
    """Test the StackedPowerLinearOperator"""
    Op = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = pylops_mpi.MPIBlockDiag(ops=[Op, ])
    FirstDeriv_MPI = pylops_mpi.MPIFirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'], )
    StackedBlockDiag_MPI = pylops_mpi.MPIStackedBlockDiag(ops=[BDiag_MPI, FirstDeriv_MPI])
    # Power Op
    Pop_MPI = StackedBlockDiag_MPI.conj()

    # For forward mode
    dist_1 = pylops_mpi.DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    dist_1[:] = np.ones(par['nx'])
    dist_2 = pylops_mpi.DistributedArray(global_shape=par['nx'] * par['ny'], dtype=par['dtype'])
    dist_2[:] = np.ones(dist_2.local_shape)
    x = pylops_mpi.StackedDistributedArray(distarrays=[dist_1, dist_2])
    x_global = x.asarray()
    Pop_x = Pop_MPI @ x
    assert isinstance(Pop_x, pylops_mpi.StackedDistributedArray)
    Pop_x_np = Pop_x.asarray()

    # For adjoint mode
    dist_1 = pylops_mpi.DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    dist_1[:] = np.ones(par['ny'])
    dist_2 = pylops_mpi.DistributedArray(global_shape=par['nx'] * par['ny'], dtype=par['dtype'])
    dist_2[:] = np.ones(dist_2.local_shape)
    y = pylops_mpi.StackedDistributedArray(distarrays=[dist_1, dist_2])
    y_global = y.asarray()
    Pop_y = Pop_MPI.H @ y
    assert isinstance(Pop_y, pylops_mpi.StackedDistributedArray)
    Pop_y_np = Pop_y.asarray()
    if rank == 0:
        ops = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in
               range(size)]
        BDiag = pylops.BlockDiag(ops=ops)
        FirstDeriv = pylops.FirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'], axis=0)
        StackedBlockDiag = pylops.BlockDiag([BDiag, FirstDeriv])
        Sop = StackedBlockDiag.conj()
        assert_allclose(Pop_x_np, Sop @ x_global, rtol=1e-14)
        assert_allclose(Pop_y_np, Sop.H @ y_global, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_sum(par):
    """Test the StackedSumLinearOperator"""
    Op1 = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI_1 = pylops_mpi.MPIBlockDiag(ops=[Op1, ])
    FirstDeriv_MPI_1 = pylops_mpi.MPIFirstDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'])
    StackedBDiag_MPI_1 = pylops_mpi.MPIStackedBlockDiag(ops=[BDiag_MPI_1, FirstDeriv_MPI_1])

    Op2 = pylops.MatrixMult(A=((rank + 2) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI_2 = pylops_mpi.MPIBlockDiag(ops=[Op2, ])
    SecondDeriv_MPI_2 = pylops_mpi.MPISecondDerivative(dims=(par['ny'], par['nx']), dtype=par['dtype'])
    StackedBDiag_MPI_2 = pylops_mpi.MPIStackedBlockDiag(ops=[BDiag_MPI_2, SecondDeriv_MPI_2])
    # Sum Op
    Sop_MPI = StackedBDiag_MPI_1 + StackedBDiag_MPI_2

    # Forward Mode
    dist1 = pylops_mpi.DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    dist1[:] = np.ones(dist1.local_shape)
    dist2 = pylops_mpi.DistributedArray(global_shape=par['ny'] * par['nx'], dtype=par['dtype'])
    dist2[:] = np.ones(dist2.local_shape)
    x = pylops_mpi.StackedDistributedArray(distarrays=[dist1, dist2])
    x_global = x.asarray()
    Sop_x = Sop_MPI @ x
    Sop_x_np = Sop_x.asarray()

    # Adjoint Mode
    dist1 = pylops_mpi.DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    dist1[:] = np.ones(dist1.local_shape)
    dist2 = pylops_mpi.DistributedArray(global_shape=par['ny'] * par['nx'], dtype=par['dtype'])
    dist2[:] = np.ones(dist2.local_shape)
    y = pylops_mpi.StackedDistributedArray(distarrays=[dist1, dist2])
    y_global = y.asarray()
    Sop_y = Sop_MPI.H @ y
    Sop_y_np = Sop_y.asarray()

    if rank == 0:
        ops1 = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in
                range(size)]
        BDiag = pylops.BlockDiag(ops=ops1)
        FirstDeriv = pylops.FirstDerivative(dims=(par['ny'], par['nx']), axis=0, dtype=par['dtype'])
        StackedBDiag_1 = pylops.BlockDiag(ops=[BDiag, FirstDeriv])
        ops2 = [pylops.MatrixMult((i + 2) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in
                range(size)]
        BDiag2 = pylops.BlockDiag(ops=ops2)
        SecondDeriv = pylops.SecondDerivative(dims=(par['ny'], par['nx']), axis=0, dtype=par['dtype'])
        StackedBDiag_2 = pylops.BlockDiag(ops=[BDiag2, SecondDeriv])
        Sop = StackedBDiag_1 + StackedBDiag_2
        assert_allclose(Sop_x_np, Sop @ x_global, rtol=1e-14)
        assert_allclose(Sop_y_np, Sop.H @ y_global, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_product(par):
    """Test the StackedProductLinearOperator"""
    Op1 = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI_1 = pylops_mpi.MPIBlockDiag(ops=[Op1, ])
    Op2 = pylops.MatrixMult(A=((rank + 2) * np.ones(shape=(par['nx'], par['ny']))).astype(par['dtype']))
    BDiag_MPI_2 = pylops_mpi.MPIBlockDiag(ops=[Op2, ])

    StackedBDiag_MPI_1 = pylops_mpi.MPIStackedBlockDiag(ops=[BDiag_MPI_1, BDiag_MPI_2])
    StackedBDiag_MPI_2 = pylops_mpi.MPIStackedBlockDiag(ops=[BDiag_MPI_2, BDiag_MPI_1])
    # Product MPI
    Pop_MPI = StackedBDiag_MPI_1 * StackedBDiag_MPI_2

    # Forward Mode
    dist1 = pylops_mpi.DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    dist1[:] = np.ones(dist1.local_shape)
    dist2 = pylops_mpi.DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    dist2[:] = np.ones(dist2.local_shape)
    x = pylops_mpi.StackedDistributedArray(distarrays=[dist1, dist2])
    x_global = x.asarray()
    Pop_x = Pop_MPI @ x
    Pop_x_np = Pop_x.asarray()

    # Adjoint Mode
    dist1 = pylops_mpi.DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    dist1[:] = np.ones(dist1.local_shape)
    dist2 = pylops_mpi.DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    dist2[:] = np.ones(dist2.local_shape)
    y = pylops_mpi.StackedDistributedArray(distarrays=[dist1, dist2])
    y_global = y.asarray()
    Pop_y = Pop_MPI.H @ y
    Pop_y_np = Pop_y.asarray()

    if rank == 0:
        ops1 = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in
                range(size)]
        BDiag1 = pylops.BlockDiag(ops=ops1)
        ops2 = [pylops.MatrixMult((i + 2) * np.ones(shape=(par['nx'], par['ny'])).astype(par['dtype'])) for i in
                range(size)]
        BDiag2 = pylops.BlockDiag(ops=ops2)
        StackedBDiag_1 = pylops.BlockDiag(ops=[BDiag1, BDiag2])
        StackedBDiag_2 = pylops.BlockDiag(ops=[BDiag2, BDiag1])
        Pop = StackedBDiag_1 * StackedBDiag_2
        assert_allclose(Pop_x_np, Pop @ x_global, rtol=1e-14)
        assert_allclose(Pop_y_np, Pop.H @ y_global, rtol=1e-14)
