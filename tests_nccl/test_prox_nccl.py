"""Test proximal operators
    Designed to run with n GPUs (with 1 MPI process per GPU)
    $ mpiexec -n 10 pytest test_prox_nccl.py --with-mpi
"""
import os

import cupy as cp
import numpy as np
import pytest
from mpi4py import MPI
from numpy.testing import assert_allclose
from pylops.basicoperators import Diagonal
from pyproximal.proximal import L0, L1, L2, L21, Box

import pylops_mpi
from pylops_mpi.proximal import MPIL2, MPIL21
from pylops_mpi.utils._nccl import initialize_nccl_comm

nccl_comm = initialize_nccl_comm()
base_comm = MPI.COMM_WORLD
size = base_comm.Get_size()
rank = base_comm.Get_rank()


par1 = {
    "n": 101,
    "imag": 0,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.SCATTER,
}  # scatter, real

par1j = {
    "n": 101,
    "imag": 1j,
    "dtype": np.complex128,
    "partition": pylops_mpi.Partition.SCATTER,
}  # scatter, complex

par1b = {
    "n": 101,
    "imag": 0,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.BROADCAST,
}  # broadcast, real


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par1b)])
def test_box(par):
    """Box call/prox vs pyproximal"""
    cp.random.seed(42)

    x = pylops_mpi.DistributedArray(
        global_shape=par["n"],
        dtype=par["dtype"],
        partition=par["partition"],
        engine="cupy",
    )
    x[:] = cp.random.normal(rank, 10, x.local_shape).astype(par["dtype"]) + par[
        "imag"
    ] * cp.random.normal(rank, 10, x.local_shape).astype(par["dtype"])
    x_global = x.asarray()

    if par["imag"] == 0:
        box = Box(lower=0.0, upper=1.0)
        boxd = pylops_mpi.proximal.MPIProxOperator(box)

        f = boxd(x)
        prox = boxd.prox(x, 0.1)
        prox = prox.asarray()

        if rank == 0:
            f_np = box(x_global)
            prox_np = box.prox(x_global, 0.1)
            assert_allclose(f, f_np, rtol=1e-14)
            assert_allclose(prox.get(), prox_np.get(), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par1b)])
def test_l0(par):
    """L0 call/prox vs pyproximal"""
    cp.random.seed(42)

    x = pylops_mpi.DistributedArray(
        global_shape=par["n"],
        dtype=par["dtype"],
        partition=par["partition"],
        engine="cupy",
    )
    x[:] = cp.random.normal(rank, 10, x.local_shape).astype(par["dtype"]) + par[
        "imag"
    ] * cp.random.normal(rank, 10, x.local_shape).astype(par["dtype"])
    x_global = x.asarray()

    l0 = L0(sigma=2.0)
    l0d = pylops_mpi.proximal.MPIProxOperator(l0)

    f = l0d(x)
    prox = l0d.prox(x, 0.1)
    prox = prox.asarray()

    if rank == 0:
        f_np = l0(x_global)
        prox_np = l0.prox(x_global, 0.1)
        assert_allclose(f, f_np, rtol=1e-14)
        assert_allclose(prox.get(), prox_np.get(), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par1b)])
def test_l1(par):
    """L1 call/prox vs pyproximal"""
    cp.random.seed(42)

    x = pylops_mpi.DistributedArray(
        global_shape=par["n"],
        dtype=par["dtype"],
        partition=par["partition"],
        engine="cupy",
    )
    x[:] = cp.random.normal(rank, 10, x.local_shape).astype(par["dtype"]) + par[
        "imag"
    ] * cp.random.normal(rank, 10, x.local_shape).astype(par["dtype"])
    x_global = x.asarray()

    l1 = L1(sigma=2.0)
    l1d = pylops_mpi.proximal.MPIProxOperator(l1)

    f = l1d(x)
    prox = l1d.prox(x, 0.1)
    prox = prox.asarray()

    if rank == 0:
        f_np = l1(x_global)
        prox_np = l1.prox(x_global, 0.1)
        assert_allclose(f, f_np, rtol=1e-14)
        assert_allclose(prox.get(), prox_np.get(), rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par1b)])
def test_precomposition(par):
    """Check precomposition method for L2 norm"""
    cp.random.seed(10)

    x = pylops_mpi.DistributedArray(
        global_shape=par["n"],
        dtype=par["dtype"],
        partition=par["partition"],
        engine="cupy",
    )
    x[:] = cp.random.normal(rank, 10, x.local_shape).astype(par["dtype"]) + par[
        "imag"
    ] * cp.random.normal(rank, 10, x.local_shape).astype(par["dtype"])

    a, b = 2.0, 5.0
    l2d = MPIL2()
    l2dprec = l2d.precomposition(a=a, b=b)

    # norm
    assert l2dprec(x) == l2d(a * x + b)

    # grad
    assert_allclose(
        l2dprec.grad(x).asarray().get(),
        a * l2d.grad(a * x + b).asarray().get(),
        rtol=1e-12,
    )

    # prox (only with b)
    l2d = MPIL2()
    l2dprec = l2d.precomposition(a=1.0, b=b)
    bd = x.zeros_like()
    bd.local_array[:] -= b
    l2dref = MPIL2(b=bd)
    assert_allclose(
        l2dprec.prox(x, 1.0).asarray().get(),
        l2dref.prox(x, 1.0).asarray().get(),
        rtol=1e-12,
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par1b)])
@pytest.mark.parametrize("solver", ["cg", "cgls"])
def test_L2(par, solver):
    """L2 proximal operator

    Test call/prox for L2 without Op/b (scatter and broadcast),
    with b (scatter and broadcast), and with Op/b (only scatter)
    """
    cp.random.seed(42)

    x = pylops_mpi.DistributedArray(
        global_shape=par["n"]
        * (size if par["partition"] == pylops_mpi.Partition.SCATTER else 1),
        dtype=par["dtype"],
        partition=par["partition"],
        engine="cupy",
    )
    x[:] = cp.random.normal(rank, 10, x.local_shape).astype(par["dtype"]) + par[
        "imag"
    ] * cp.random.normal(rank, 10, x.local_shape).astype(par["dtype"])
    x_global = x.asarray()

    b = pylops_mpi.DistributedArray(
        global_shape=par["n"]
        * (size if par["partition"] == pylops_mpi.Partition.SCATTER else 1),
        dtype=par["dtype"],
        partition=par["partition"],
        engine="cupy",
    )
    b[:] = cp.random.normal(rank, 10, x.local_shape).astype(par["dtype"]) + par[
        "imag"
    ] * cp.random.normal(rank, 10, x.local_shape).astype(par["dtype"])
    b_global = b.asarray()

    Op_global = Diagonal(
        cp.ones(
            par["n"]
            * (size if par["partition"] == pylops_mpi.Partition.SCATTER else 1),
            dtype=par["dtype"],
        ),
        dtype=par["dtype"],
    )
    if par["partition"] == pylops_mpi.Partition.SCATTER:
        Op_local = Diagonal(cp.ones(par["n"], dtype=par["dtype"]), dtype=par["dtype"])
        Opd = pylops_mpi.MPIBlockDiag(
            [
                Op_local,
            ]
        )
    else:
        Op_local = Diagonal(cp.ones(par["n"], dtype=par["dtype"]), dtype=par["dtype"])
        Opd = pylops_mpi.MPILinearOperator(Op_local)

    l2x = L2(sigma=2.0)
    l2xd = MPIL2(sigma=2.0)

    l2b = L2(b=b_global, sigma=2.0)
    l2bd = MPIL2(b=b, sigma=2.0)

    l2Op = L2(Op=Op_global, b=b_global, sigma=2.0, solver=solver)
    l2Opd = MPIL2(Op=Opd, b=b, sigma=2.0, x0=x.zeros_like(), solver=solver)

    for l2, l2d in zip([l2x, l2b, l2Op], [l2xd, l2bd, l2Opd]):
        f = l2d(x)
        prox = l2d.prox(x, 0.1)
        prox = prox.asarray()

        if rank == 0:
            f_np = l2(x_global)
            prox_np = l2.prox(x_global, 0.1)
            assert_allclose(f, f_np, rtol=1e-14)
            assert_allclose(prox.get(), prox_np.get(), rtol=1e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par",
    [
        (par1),
    ],
)
def test_l21_wrongdim(par):
    """L21 raises an error if ndim is not the same of the number of
    DistributedArray input StackedDistributedArray"""
    cp.random.seed(42)

    ndim = 3
    x = []
    for _ in range(ndim - 1):
        x_ = pylops_mpi.DistributedArray(
            global_shape=par["n"],
            dtype=par["dtype"],
            partition=par["partition"],
            engine="cupy",
        )
        x_[:] = cp.random.normal(rank, 10, x_.local_shape).astype(par["dtype"])
        x.append(x_)
    x = pylops_mpi.StackedDistributedArray(x)

    l21d = MPIL21(ndim=ndim, sigma=2.0)

    with pytest.raises(ValueError, match=f"Expected {ndim} DistributedArray"):
        _ = l21d(x)

    with pytest.raises(ValueError, match=f"Expected {ndim} DistributedArray"):
        _ = l21d.prox(x, 0.1)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "par", [(par1), (par1b)]
)  # (par1j) excluded until PyProximal #257 is merged
def test_l21(par):
    """L21 call/prox vs pyproximal"""
    cp.random.seed(42)

    ndim = 3
    x = []
    x_global = cp.zeros((ndim, par["n"]), dtype=par["dtype"]) + par["imag"] * cp.zeros(
        (ndim, par["n"]), dtype=par["dtype"]
    )
    for i in range(ndim):
        x_ = pylops_mpi.DistributedArray(
            global_shape=par["n"],
            dtype=par["dtype"],
            partition=par["partition"],
            engine="cupy",
        )
        x_[:] = cp.random.normal(rank, 10, x_.local_shape).astype(par["dtype"]) + par[
            "imag"
        ] * cp.random.normal(rank, 10, x_.local_shape).astype(par["dtype"])
        x.append(x_)
        x_global[i] = x_.asarray()
    x = pylops_mpi.StackedDistributedArray(x)
    x_global = x_global.flatten()

    l21 = L21(ndim=ndim, sigma=2.0)
    l21d = MPIL21(ndim=ndim, sigma=2.0)

    f = l21d(x)
    prox = l21d.prox(x, 0.1)
    prox = prox.asarray()
    proxd = l21d.proxdual(x, 0.1)
    proxd = proxd.asarray()

    if rank == 0:
        f_np = l21(x_global)
        prox_np = l21.prox(x_global, 0.1)
        proxd_np = l21.proxdual(x_global, 0.1)
        assert_allclose(f, f_np, rtol=1e-14)
        assert_allclose(prox.get(), prox_np.get(), rtol=1e-14)
        assert_allclose(proxd.get(), proxd_np.get(), rtol=1e-14)
