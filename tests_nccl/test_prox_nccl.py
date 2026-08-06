"""Test proximal operators
    Designed to run with n GPUs (with 1 MPI process per GPU)
    $ mpiexec -n 10 pytest test_prox_nccl.py --with-mpi
"""
import os

import numpy as np
import cupy as cp
from numpy.testing import assert_allclose
from mpi4py import MPI
import pytest

import pylops_mpi
from pylops.basicoperators import Diagonal
from pyproximal.proximal import (
    Box,
    L0,
    L1,
    L2
)
from pylops_mpi.proximal import MPIL2
from pylops_mpi.utils._nccl import initialize_nccl_comm


nccl_comm = initialize_nccl_comm()
base_comm = MPI.COMM_WORLD
size = base_comm.Get_size()
rank = base_comm.Get_rank()


par1 = {
    "n": 101,
    "imag": 0,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.SCATTER
}  # scatter, real

par1j = {
    "n": 101,
    "imag": 1j,
    "dtype": np.complex128,
    "partition": pylops_mpi.Partition.SCATTER
}  # scatter, complex

par1b = {
    "n": 101,
    "imag": 0,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.BROADCAST
}  # broadcast, real


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par1b)]
)
def test_box(par):
    """Box call/prox vs pyproximal"""
    cp.random.seed(42)

    x = pylops_mpi.DistributedArray(global_shape=par['n'], dtype=par['dtype'],
                                    partition=par['partition'], engine="cupy")
    x[:] = cp.random.normal(rank, 10, x.local_shape).astype(par['dtype']) + \
        par['imag'] * cp.random.normal(rank, 10, x.local_shape).astype(par['dtype'])
    x_global = x.asarray()

    if par['imag'] == 0:
        box = Box(lower=0.0, upper=1.0)
        boxd = pylops_mpi.proximal.MPIProxOperator(box)

        f = boxd(x)
        prox = boxd.prox(x, .1)
        prox = prox.asarray()

        if rank == 0:
            f_np = box(x_global)
            prox_np = box.prox(x_global, .1)
            assert_allclose(f, f_np, rtol=1e-14)
            assert_allclose(prox.get(), prox_np.get(), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par1b)]
)
def test_l0(par):
    """L0 call/prox vs pyproximal"""
    cp.random.seed(42)

    x = pylops_mpi.DistributedArray(global_shape=par['n'], dtype=par['dtype'],
                                    partition=par['partition'], engine="cupy")
    x[:] = cp.random.normal(rank, 10, x.local_shape).astype(par['dtype']) + \
        par['imag'] * cp.random.normal(rank, 10, x.local_shape).astype(par['dtype'])
    x_global = x.asarray()

    l0 = L0(sigma=2.0)
    l0d = pylops_mpi.proximal.MPIProxOperator(l0)

    f = l0d(x)
    prox = l0d.prox(x, .1)
    prox = prox.asarray()

    if rank == 0:
        f_np = l0(x_global)
        prox_np = l0.prox(x_global, .1)
        assert_allclose(f, f_np, rtol=1e-14)
        assert_allclose(prox.get(), prox_np.get(), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1j), (par1b)]
)
def test_l1(par):
    """L1 call/prox vs pyproximal"""
    cp.random.seed(42)

    x = pylops_mpi.DistributedArray(global_shape=par['n'], dtype=par['dtype'],
                                    partition=par['partition'], engine="cupy")
    x[:] = cp.random.normal(rank, 10, x.local_shape).astype(par['dtype']) + \
        par['imag'] * cp.random.normal(rank, 10, x.local_shape).astype(par['dtype'])
    x_global = x.asarray()

    l1 = L1(sigma=2.0)
    l1d = pylops_mpi.proximal.MPIProxOperator(l1)

    f = l1d(x)
    prox = l1d.prox(x, .1)
    prox = prox.asarray()

    if rank == 0:
        f_np = l1(x_global)
        prox_np = l1.prox(x_global, .1)
        # assert_allclose(f, f_np, rtol=1e-14)
        assert_allclose(prox.get(), prox_np.get(), rtol=1e-14)
