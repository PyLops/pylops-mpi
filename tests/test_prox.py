"""Test proximal operators
    Designed to run with n processes
    $ mpiexec -n 10 pytest test_prox.py --with-mpi
"""
import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_allclose

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_allclose

    backend = "numpy"
from mpi4py import MPI
import pytest

import pylops_mpi
from pylops.basicoperators import FirstDerivative
from pyproximal.proximal import (
    Box,
    L0,
    L1,
    L2
)
from pylops_mpi.proximal import MPIL2


size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
if backend == "cupy":
    device_id = rank % np.cuda.runtime.getDeviceCount()
    np.cuda.Device(device_id).use()


par1 = {
    "n": 101,
    "imag": 0,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.SCATTER
}

par1b = {
    "n": 101,
    "imag": 0,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.BROADCAST
}

par1j = {
    "n": 101,
    "imag": 1j,
    "dtype": np.complex128,
    "partition": pylops_mpi.Partition.SCATTER
}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1b), (par1j),]
)
def test_separable_prox(par):
    """Separable proximal operators"""
    np.random.seed(42)

    x = pylops_mpi.DistributedArray(global_shape=par['n'], dtype=par['dtype'],
                                    partition=par['partition'], engine=backend)
    x[:] = np.random.normal(rank, 10, x.local_shape).astype(par['dtype']) + \
        par['imag'] * np.random.normal(rank, 10, x.local_shape).astype(par['dtype'])
    x_global = x.asarray()

    # Box (does not support complex numbers)
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
            assert_allclose(prox, prox_np, rtol=1e-14)

    # L0
    l0 = L0(sigma=2.0)
    l0d = pylops_mpi.proximal.MPIProxOperator(l0)

    f = l0d(x)
    prox = l0d.prox(x, .1)
    prox = prox.asarray()

    if rank == 0:
        f_np = l0(x_global)
        prox_np = l0.prox(x_global, .1)
        assert_allclose(f, f_np, rtol=1e-14)
        assert_allclose(prox, prox_np, rtol=1e-14)

    # L1
    l1 = L1(sigma=2.0)
    l1d = pylops_mpi.proximal.MPIProxOperator(l1)

    f = l1d(x)
    prox = l1d.prox(x, .1)
    prox = prox.asarray()

    if rank == 0:
        f_np = l1(x_global)
        prox_np = l1.prox(x_global, .1)
        assert_allclose(f, f_np, rtol=1e-14)
        assert_allclose(prox, prox_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1b), (par1j),]
)
def test_L2(par):
    """L2 proximal operator"""
    np.random.seed(42)

    x = pylops_mpi.DistributedArray(global_shape=par['n'], dtype=par['dtype'],
                                    partition=par['partition'], engine=backend)
    x[:] = np.random.normal(rank, 10, x.local_shape).astype(par['dtype']) + \
        par['imag'] * np.random.normal(rank, 10, x.local_shape).astype(par['dtype'])
    x_global = x.asarray()

    b = pylops_mpi.DistributedArray(global_shape=par['n'], dtype=par['dtype'],
                                    partition=par['partition'], engine=backend)
    b[:] = np.random.normal(rank, 10, x.local_shape).astype(par['dtype']) + \
        par['imag'] * np.random.normal(rank, 10, x.local_shape).astype(par['dtype'])
    b_global = b.asarray()

    Op_global = FirstDerivative(
        par['n'] * (size if par["partition"] == pylops_mpi.Partition.SCATTER else 1),
        sampling=0.001)
    Opd = pylops_mpi.MPIFirstDerivative(
        par['n'] * (size if par["partition"] == pylops_mpi.Partition.SCATTER else 1),
        sampling=0.001)

    l2x = L2(sigma=2.0)
    l2xd = MPIL2(sigma=2.0)

    l2b = L2(b=b_global, sigma=2.0)
    l2bd = MPIL2(b=b, sigma=2.0)

    # l2Op = L2(Op=Op_global, b=b_global, sigma=2.0)
    # l2Opd = MPIL2(Op=Opd, b=b, sigma=2.0)

    # for l2, l2d in zip([l2x, l2b, l2Op], [l2xd, l2bd, l2Opd]):
    for l2, l2d in zip([l2x, l2b,], [l2xd, l2bd,]):
        f = l2d(x)
        prox = l2d.prox(x, .1)
        prox = prox.asarray()

        if rank == 0:
            f_np = l2(x_global)
            prox_np = l2.prox(x_global, .1)
            assert_allclose(f, f_np, rtol=1e-14)
            assert_allclose(prox, prox_np, rtol=1e-14)
