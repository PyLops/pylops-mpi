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
import pytest
from mpi4py import MPI
from pylops.basicoperators import Diagonal
from pyproximal.proximal import L0, L1, L2, L21, Box, Simplex
from pyproximal.proximal import VStack as ProxVStack

import pylops_mpi
from pylops_mpi.proximal import MPIL2, MPIL21, MPISimplex
from pylops_mpi.proximal.proximal.VStack import MPIProxStackedVStack

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
if backend == "cupy":
    device_id = rank % np.cuda.runtime.getDeviceCount()
    np.cuda.Device(device_id).use()


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
@pytest.mark.parametrize(
    "proxop",
    [Box(lower=0.0, upper=1.0), L0(sigma=2.0), L1(sigma=2.0)],
    ids=["box", "l0", "l1"],
)
@pytest.mark.parametrize("stacked", [False, True])
def test_separable(par, proxop, stacked):
    """Separable prox (Box, L0, L1) call/prox vs pyproximal,
    for DistributedArray and StackedDistributedArray"""
    np.random.seed(42)

    narrays = 3 if stacked else 1
    xs = []
    for _ in range(narrays):
        x_ = pylops_mpi.DistributedArray(
            global_shape=par["n"],
            dtype=par["dtype"],
            partition=par["partition"],
            engine=backend,
        )
        x_[:] = np.random.normal(rank, 10, x_.local_shape).astype(par["dtype"]) + par[
            "imag"
        ] * np.random.normal(rank, 10, x_.local_shape).astype(par["dtype"])
        xs.append(x_)
    x = pylops_mpi.StackedDistributedArray(xs) if stacked else xs[0]
    x_global = np.concatenate([xi.asarray() for xi in xs])

    proxd = pylops_mpi.proximal.MPIProxOperator(proxop)
    f = proxd(x)
    prox = proxd.prox(x, 0.1).asarray()

    if rank == 0:
        f_np = proxop(x_global)
        prox_np = proxop.prox(x_global, 0.1)
        assert_allclose(f, f_np, rtol=1e-14)
        assert_allclose(prox, prox_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par1b)])
def test_postcomposition(par):
    """Check postcomposition method for L2 norm"""
    np.random.seed(10)

    x = pylops_mpi.DistributedArray(
        global_shape=par["n"],
        dtype=par["dtype"],
        partition=par["partition"],
        engine=backend,
    )
    x[:] = np.random.normal(rank, 10, x.local_shape).astype(par["dtype"]) + par[
        "imag"
    ] * np.random.normal(rank, 10, x.local_shape).astype(par["dtype"])

    sigma = 2.0
    l2d = MPIL2()
    l2dpostc = l2d.postcomposition(sigma=sigma)

    # norm
    assert l2dpostc(x) == sigma * l2d(x)

    # grad
    assert_allclose(
        l2dpostc.grad(x).asarray(),
        sigma * l2d.grad(x).asarray(),
        rtol=1e-12,
    )

    # prox
    assert_allclose(
        l2dpostc.prox(x, 1.0).asarray(),
        l2d.prox(x, sigma).asarray(),
        rtol=1e-12,
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par1b)])
def test_precomposition(par):
    """Check precomposition method for L2 norm"""
    np.random.seed(10)

    x = pylops_mpi.DistributedArray(
        global_shape=par["n"],
        dtype=par["dtype"],
        partition=par["partition"],
        engine=backend,
    )
    x[:] = np.random.normal(rank, 10, x.local_shape).astype(par["dtype"]) + par[
        "imag"
    ] * np.random.normal(rank, 10, x.local_shape).astype(par["dtype"])

    a, b = 2.0, 5.0
    l2d = MPIL2()
    l2dprec = l2d.precomposition(a=a, b=b)

    # norm
    assert l2dprec(x) == l2d(a * x + b)

    # grad
    assert_allclose(
        l2dprec.grad(x).asarray(),
        a * l2d.grad(a * x + b).asarray(),
        rtol=1e-12,
    )

    # prox / proxdual (only with b)
    l2d = MPIL2()
    l2dprec = l2d.precomposition(a=1.0, b=b)
    bd = x.zeros_like()
    bd.local_array[:] -= b
    l2dref = MPIL2(b=bd)

    assert_allclose(
        l2dprec.prox(x, 1.0).asarray(),
        l2dref.prox(x, 1.0).asarray(),
        rtol=1e-12,
    )
    assert_allclose(
        l2dprec.proxdual(x, 1.0).asarray(),
        l2dref.proxdual(x, 1.0).asarray(),
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
    np.random.seed(42)

    x = pylops_mpi.DistributedArray(
        global_shape=par["n"]
        * (size if par["partition"] == pylops_mpi.Partition.SCATTER else 1),
        dtype=par["dtype"],
        partition=par["partition"],
        engine=backend,
    )
    x[:] = np.random.normal(rank, 10, x.local_shape).astype(par["dtype"]) + par[
        "imag"
    ] * np.random.normal(rank, 10, x.local_shape).astype(par["dtype"])
    x_global = x.asarray()

    b = pylops_mpi.DistributedArray(
        global_shape=par["n"]
        * (size if par["partition"] == pylops_mpi.Partition.SCATTER else 1),
        dtype=par["dtype"],
        partition=par["partition"],
        engine=backend,
    )
    b[:] = np.random.normal(rank, 10, x.local_shape).astype(par["dtype"]) + par[
        "imag"
    ] * np.random.normal(rank, 10, x.local_shape).astype(par["dtype"])
    b_global = b.asarray()

    Op_global = Diagonal(
        np.ones(
            par["n"]
            * (size if par["partition"] == pylops_mpi.Partition.SCATTER else 1),
            dtype=par["dtype"],
        ),
        dtype=par["dtype"],
    )
    if par["partition"] == pylops_mpi.Partition.SCATTER:
        Op_local = Diagonal(np.ones(par["n"], dtype=par["dtype"]), dtype=par["dtype"])
        Opd = pylops_mpi.MPIBlockDiag(
            [
                Op_local,
            ]
        )
    else:
        Op_local = Diagonal(np.ones(par["n"], dtype=par["dtype"]), dtype=par["dtype"])
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
            assert_allclose(prox, prox_np, rtol=1e-12)


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
    np.random.seed(42)

    ndim = 3
    x = []
    for _ in range(ndim - 1):
        x_ = pylops_mpi.DistributedArray(
            global_shape=par["n"],
            dtype=par["dtype"],
            partition=par["partition"],
            engine=backend,
        )
        x_[:] = np.random.normal(rank, 10, x_.local_shape).astype(par["dtype"])
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
    np.random.seed(42)

    ndim = 3
    x = []
    x_global = np.zeros((ndim, par["n"]), dtype=par["dtype"]) + par["imag"] * np.zeros(
        (ndim, par["n"]), dtype=par["dtype"]
    )
    for i in range(ndim):
        x_ = pylops_mpi.DistributedArray(
            global_shape=par["n"],
            dtype=par["dtype"],
            partition=par["partition"],
            engine=backend,
        )
        x_[:] = np.random.normal(rank, 10, x_.local_shape).astype(par["dtype"]) + par[
            "imag"
        ] * np.random.normal(rank, 10, x_.local_shape).astype(par["dtype"])
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
        assert_allclose(prox, prox_np, rtol=1e-14)
        assert_allclose(proxd, proxd_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1b)])
def test_simplex(par):
    """Simplex prox vs pyproximal"""
    np.random.seed(42)

    ndim = 3
    radius = 1.0
    x = []
    for _ in range(ndim):
        x_ = pylops_mpi.DistributedArray(
            global_shape=par["n"],
            dtype=par["dtype"],
            partition=par["partition"],
            engine=backend,
        )
        x_[:] = np.random.normal(rank, 10, x_.local_shape).astype(par["dtype"])
        x.append(x_)
    x = pylops_mpi.StackedDistributedArray(x)

    # gather input for local comparison, shape (ndim, n)
    x_global = np.vstack([xi.asarray() for xi in x])

    # each column (same index across the ndim arrays) is
    # projected onto the simplex
    nloc = x[0].local_shape[0]
    simplexd = MPISimplex(
        dims=(ndim, nloc),
        radius=radius,
        engine="numba" if backend == "numpy" else "cuda",
    )

    prox = simplexd.prox(x, 0.1)
    c_in = simplexd(x)
    c_out = simplexd(prox, tol=1e-6)
    prox = prox.asarray()
    assert not c_in
    assert c_out

    if rank == 0:
        simplex = Simplex(
            ndim * par["n"],
            radius,
            dims=(ndim, par["n"]),
            axis=0,
            engine="numba" if backend == "numpy" else "cuda",
        )
        prox_np = simplex.prox(x_global.ravel(), 0.1)
        # tolerances selected in line with ftol/xtol in Simplex
        assert_allclose(prox, prox_np, rtol=1e-6, atol=1e-7)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par1b)])
@pytest.mark.parametrize("proxop", [Box, L2, L1, L0])
@pytest.mark.parametrize("nstack", [3, 4])
def test_vstack(par, proxop, nstack):
    """VStack call/prox vs pyproximal"""
    np.random.seed(42)

    x = pylops_mpi.DistributedArray(
        global_shape=par["n"]
        * (size if par["partition"] == pylops_mpi.Partition.SCATTER else 1),
        dtype=par["dtype"],
        partition=par["partition"],
        engine=backend,
    )
    x[:] = np.random.normal(rank, 10, x.local_shape).astype(par["dtype"]) + par[
        "imag"
    ] * np.random.normal(rank, 10, x.local_shape).astype(par["dtype"])
    x_global = x.asarray()

    nlocal = x.local_shape[0]
    ns = [nlocal // nstack] * nstack
    ns[-1] = nlocal - np.sum(ns[:-1])
    nsg = comm.allgather(ns)
    nsg = [n for ns in nsg for n in ns]  # flatten list

    pop = proxop()
    pvstack = ProxVStack([pop] * nstack, ns)
    pvstackd = pylops_mpi.proximal.MPIProxOperator(pvstack)

    f = pvstackd(x)
    prox = pvstackd.prox(x, 0.1)
    prox = prox.asarray()

    if rank == 0:
        pvstack = ProxVStack(
            [pop]
            * nstack
            * (size if par["partition"] == pylops_mpi.Partition.SCATTER else 1),
            nsg if par["partition"] == pylops_mpi.Partition.SCATTER else ns,
        )

        f_np = pvstack(x_global)
        prox_np = pvstack.prox(x_global, 0.1)
        assert_allclose(f, f_np, rtol=1e-14)
        assert_allclose(prox, prox_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par1b)])
def test_stackedvstack(par):
    """MPIProxStackedVStack call/prox vs pyproximal applied to each
    element of the StackedDistributedArray"""
    np.random.seed(42)

    proxops = [L1(sigma=2.0), L0(sigma=1.0), L1(sigma=0.5)]
    xs = []
    for _ in proxops:
        x_ = pylops_mpi.DistributedArray(
            global_shape=par["n"],
            dtype=par["dtype"],
            partition=par["partition"],
            engine=backend,
        )
        x_[:] = np.random.normal(rank, 10, x_.local_shape).astype(par["dtype"]) + par[
            "imag"
        ] * np.random.normal(rank, 10, x_.local_shape).astype(par["dtype"])
        xs.append(x_)
    x = pylops_mpi.StackedDistributedArray(xs)
    x_global = np.concatenate([xi.asarray() for xi in xs])
    nsg = [xi.global_shape[0] for xi in xs]

    vstackd = MPIProxStackedVStack(
        [pylops_mpi.proximal.MPIProxOperator(op) for op in proxops]
    )
    f = vstackd(x)
    prox = vstackd.prox(x, 0.1)
    assert isinstance(prox, pylops_mpi.StackedDistributedArray)
    assert prox.narrays == len(proxops)
    prox = np.concatenate([p.asarray() for p in prox])

    if rank == 0:
        pvstack = ProxVStack(proxops, nsg)
        f_np = pvstack(x_global)
        prox_np = pvstack.prox(x_global, 0.1)
        assert_allclose(f, f_np, rtol=1e-14)
        assert_allclose(prox, prox_np, rtol=1e-14)
