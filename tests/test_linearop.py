import numpy as np
from numpy.testing import assert_allclose
np.random.seed(42)
import pytest

from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

import pylops

from pylops_mpi import (
    asmpilinearoperator,
    DistributedArray,
    MPILinearOperator,
    MPIBlockDiag,
    MPIVStack,
    Partition
)

par1 = {'ny': 101, 'nx': 101, 'dtype': np.float64}
par1j = {'ny': 101, 'nx': 101, 'dtype': np.complex128}
par2 = {'ny': 301, 'nx': 101, 'dtype': np.float64}
par2j = {'ny': 301, 'nx': 101, 'dtype': np.complex128}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_linearop(par):
    """Apply various overloaded operators (.H, .T, +, *, conj()) and ensure that the
    returned operator is still of `pylops_mpi.MPILinearOperator` type
    """
    Op = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = MPIBlockDiag(ops=[Op, ])
    assert isinstance(BDiag_MPI, MPILinearOperator)
    assert isinstance(BDiag_MPI.H, MPILinearOperator)
    assert isinstance(BDiag_MPI.T, MPILinearOperator)
    assert isinstance(BDiag_MPI * -3, MPILinearOperator)
    assert isinstance(BDiag_MPI.conj(), MPILinearOperator)
    assert isinstance(BDiag_MPI + BDiag_MPI, MPILinearOperator)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_square_linearop(par):
    """Apply overloaded operators (**, *) to square operators and
    ensure that the returned operator is still of `pylops_mpi.MPILinearOperator` type
    """
    Op = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = MPIBlockDiag(ops=[Op, ])
    assert isinstance(BDiag_MPI ** 4, MPILinearOperator)
    assert isinstance(BDiag_MPI * BDiag_MPI, MPILinearOperator)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_transpose(par):
    """Test the TransposeLinearOperator"""
    Op = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = MPIBlockDiag(ops=[Op, ])
    # Tranposed Op
    Top_MPI = BDiag_MPI.T

    # For forward mode
    x = DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    x[:] = np.ones(par['ny'])
    x_global = x.asarray()
    Top_x = Top_MPI @ x
    assert isinstance(Top_x, DistributedArray)
    Top_x_np = Top_x.asarray()

    # For adjoint mode
    y = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    y[:] = np.ones(par['nx'])
    y_global = y.asarray()
    Top_y = Top_MPI.H @ y
    assert isinstance(Top_y, DistributedArray)
    Top_y_np = Top_y.asarray()
    if rank == 0:
        ops = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in
               range(size)]
        BDiag = pylops.BlockDiag(ops=ops)
        Top = BDiag.T
        assert_allclose(Top_x_np, Top @ x_global, rtol=1e-14)
        assert_allclose(Top_y_np, Top.H @ y_global, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_scaled(par):
    """Test the ScaledLinearOperator"""
    Op = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = MPIBlockDiag(ops=[Op, ])
    # Scaled Op
    Sop_MPI = BDiag_MPI * -4

    # For forward mode
    x = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    x[:] = np.ones(par['nx'])
    x_global = x.asarray()
    Sop_x = Sop_MPI @ x
    assert isinstance(Sop_x, DistributedArray)
    Sop_x_np = Sop_x.asarray()

    # For adjoint mode
    y = DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    y[:] = np.ones(par['ny'])
    y_global = y.asarray()
    Sop_y = Sop_MPI.H @ y
    assert isinstance(Sop_y, DistributedArray)
    Sop_y_np = Sop_y.asarray()
    if rank == 0:
        ops = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in
               range(size)]
        BDiag = pylops.BlockDiag(ops=ops)
        Sop = BDiag * -4
        assert_allclose(Sop_x_np, Sop @ x_global, rtol=1e-14)
        assert_allclose(Sop_y_np, Sop.H @ y_global, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_power(par):
    """Test the PowerLinearOperator"""
    Op = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = MPIBlockDiag(ops=[Op, ])
    # Power Operator
    Pop_MPI = BDiag_MPI ** 3

    # Forward Mode
    x = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    x[:] = np.ones(par['nx'])
    x_global = x.asarray()
    Pop_x = Pop_MPI @ x
    assert isinstance(Pop_x, DistributedArray)
    Pop_x_np = Pop_x.asarray()

    # Adjoint Mode
    y = DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    y[:] = np.ones(par['ny'])
    y_global = y.asarray()
    Pop_y = Pop_MPI.H @ y
    assert isinstance(Pop_y, DistributedArray)
    Pop_y_np = Pop_y.asarray()

    if rank == 0:
        ops = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in
               range(size)]
        BDiag = pylops.BlockDiag(ops=ops)
        Pop = BDiag ** 3
        assert_allclose(Pop_x_np, Pop @ x_global, rtol=1e-14)
        assert_allclose(Pop_y_np, Pop.H @ y_global, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_sum(par):
    """Test the SumLinearOperator"""
    Op1 = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI_1 = MPIBlockDiag(ops=[Op1, ])

    Op2 = pylops.MatrixMult(A=((rank + 2) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI_2 = MPIBlockDiag(ops=[Op2, ])
    # Sum Op
    Sop_MPI = BDiag_MPI_1 + BDiag_MPI_2

    # Forward Mode
    x = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    x[:] = np.ones(par['nx'])
    x_global = x.asarray()
    Sop_x = Sop_MPI @ x
    assert isinstance(Sop_x, DistributedArray)
    Sop_x_np = Sop_x.asarray()

    # Adjoint Mode
    y = DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    y[:] = np.ones(par['ny'])
    y_global = y.asarray()
    Sop_y = Sop_MPI.H @ y
    assert isinstance(Sop_y, DistributedArray)
    Sop_y_np = Sop_y.asarray()

    if rank == 0:
        ops1 = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in
                range(size)]
        BDiag = pylops.BlockDiag(ops=ops1)
        ops2 = [pylops.MatrixMult((i + 2) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in
                range(size)]
        BDiag2 = pylops.BlockDiag(ops=ops2)
        Sop = BDiag + BDiag2
        assert_allclose(Sop_x_np, Sop @ x_global, rtol=1e-14)
        assert_allclose(Sop_y_np, Sop.H @ y_global, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_product(par):
    """Test ProductLinearOperator"""
    Op1 = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI_1 = MPIBlockDiag(ops=[Op1, ])

    Op2 = pylops.MatrixMult(A=((rank + 2) * np.ones(shape=(par['nx'], par['ny']))).astype(par['dtype']))
    BDiag_MPI_2 = MPIBlockDiag(ops=[Op2, ])
    # Product Op
    Pop_MPI = BDiag_MPI_1 * BDiag_MPI_2

    # Forward Mode
    x = DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    x[:] = np.ones(par['ny'])
    x_global = x.asarray()
    Pop_x = Pop_MPI @ x
    assert isinstance(Pop_x, DistributedArray)
    Pop_x_np = Pop_x.asarray()

    # Adjoint Mode
    y = DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    y[:] = np.ones(par['ny'])
    y_global = y.asarray()
    Pop_y = Pop_MPI.H @ y
    assert isinstance(Pop_y, DistributedArray)
    Pop_y_np = Pop_y.asarray()

    if rank == 0:
        ops1 = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in
                range(size)]
        BDiag = pylops.BlockDiag(ops=ops1)
        ops2 = [pylops.MatrixMult((i + 2) * np.ones(shape=(par['nx'], par['ny'])).astype(par['dtype'])) for i in
                range(size)]
        BDiag2 = pylops.BlockDiag(ops=ops2)
        Pop = BDiag * BDiag2
        assert_allclose(Pop_x_np, Pop @ x_global, rtol=1e-14)
        assert_allclose(Pop_y_np, Pop.H @ y_global, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_conj(par):
    """Test the ConjLinearOperator"""
    Op = pylops.MatrixMult(A=((rank + 1) * np.ones(shape=(par['ny'], par['nx']))).astype(par['dtype']))
    BDiag_MPI = MPIBlockDiag(ops=[Op, ])
    # Conj Op
    Cop_MPI = BDiag_MPI.conj()

    # For forward mode
    x = DistributedArray(global_shape=size * par['nx'], dtype=par['dtype'])
    x[:] = np.ones(par['nx'])
    x_global = x.asarray()
    Cop_x = Cop_MPI @ x
    assert isinstance(Cop_x, DistributedArray)
    Cop_x_np = Cop_x.asarray()

    # For adjoint mode
    y = DistributedArray(global_shape=size * par['ny'], dtype=par['dtype'])
    y[:] = np.ones(par['ny'])
    y_global = y.asarray()
    Cop_y = Cop_MPI.H @ y
    assert isinstance(Cop_y, DistributedArray)
    Cop_y_np = Cop_y.asarray()

    if rank == 0:
        ops = [pylops.MatrixMult((i + 1) * np.ones(shape=(par['ny'], par['nx'])).astype(par['dtype'])) for i in
               range(size)]
        BDiag = pylops.BlockDiag(ops=ops)
        Cop = BDiag.conj()
        assert_allclose(Cop_x_np, Cop @ x_global, rtol=1e-14)
        assert_allclose(Cop_y_np, Cop.H @ y_global, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_mpilinop(par):
    """Wrapping the Pylops Linear Operators"""
    Fop = pylops.FirstDerivative(dims=(par['ny'], par['nx']), axis=0, dtype=par['dtype'])
    Mop = asmpilinearoperator(Op=Fop)
    assert isinstance(Mop, MPILinearOperator)
    # DistributedArray with partition="BROADCAST"
    x = DistributedArray(global_shape=Mop.shape[1],
                         partition=Partition.BROADCAST, dtype=par['dtype'])
    x[:] = np.random.normal(1, 10, x.local_shape).astype(par['dtype'])
    x_global = x.asarray()
    # Test for single MPILinearOperator
    y_dist = Mop @ x
    y = y_dist.asarray()

    # Test Product of MPIVStack and MPILinearOperator
    Sop = pylops.SecondDerivative(dims=(par['ny'], par['nx']), axis=0, dtype=par['dtype'])
    VStack_MPI = MPIVStack(ops=[(rank + 1) * Sop, ])
    FullOp_MPI = VStack_MPI @ Mop
    # Forward
    y_full_dist = FullOp_MPI @ x
    y_full = y_full_dist.asarray()

    # Adjoint
    # DistributedArray with partition="SCATTER"
    x_adj = DistributedArray(global_shape=FullOp_MPI.shape[0],
                             partition=Partition.SCATTER, dtype=par['dtype'])
    x_adj[:] = np.random.normal(0, 1, x.local_shape).astype(par['dtype'])
    x_adj_global = x_adj.asarray()
    y_adj_dist = FullOp_MPI.H @ x_adj
    y_adj = y_adj_dist.asarray()

    if rank == 0:
        assert_allclose(y, Fop @ x_global, rtol=1e-14)
        VStack = pylops.VStack(ops=[(i + 1) * Sop for i in range(size)])
        FullOp = VStack @ Fop
        y_np = FullOp @ x_global
        y_adj_np = FullOp.H @ x_adj_global
        assert_allclose(y_full, y_np, rtol=1e-14)
        assert_allclose(y_adj, y_adj_np.flatten(), rtol=1e-14)
