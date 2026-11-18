"""Test the MPIFredholm1 class
    Designed to run with n processes
    $ mpiexec -n 10 pytest test_fredholm.py --with-mpi
"""
import math
import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_allclose

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_allclose

    backend = "numpy"
import numpy as npp
from mpi4py import MPI
import pytest

import pylops
import pylops_mpi

from pylops_mpi import DistributedArray
from pylops_mpi.DistributedArray import local_split, Partition
from pylops_mpi.signalprocessing import MPIFredholm1, MPIFredholm1SUMMA
from pylops_mpi.utils.dottest import dottest

np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
if backend == "cupy":
    device_id = rank % np.cuda.runtime.getDeviceCount()
    np.cuda.Device(device_id).use()

par1 = {
    "nsl": 21,
    "ny": 6,
    "nx": 4,
    "nz": 5,
    "usematmul": False,
    "saveGt": True,
    "imag": 0,
    "dtype": "float32",
}  # real, saved Gt
par2 = {
    "nsl": 21,
    "ny": 6,
    "nx": 4,
    "nz": 5,
    "usematmul": True,
    "saveGt": False,
    "imag": 0,
    "dtype": "float32",
}  # real, unsaved Gt
par3 = {
    "nsl": 21,
    "ny": 6,
    "nx": 4,
    "nz": 5,
    "usematmul": False,
    "saveGt": True,
    "imag": 1j,
    "dtype": "complex64",
}  # complex, saved Gt
par4 = {
    "nsl": 21,
    "ny": 6,
    "nx": 4,
    "nz": 5,
    "saveGt": False,
    "usematmul": False,
    "imag": 1j,
    "dtype": "complex64",
}  # complex, unsaved Gt
par5 = {
    "nsl": 21,
    "ny": 6,
    "nx": 4,
    "nz": 1,
    "usematmul": True,
    "saveGt": True,
    "imag": 0,
    "dtype": "float32",
}  # real, saved Gt, nz=1
par6 = {
    "nsl": 21,
    "ny": 6,
    "nx": 4,
    "nz": 1,
    "usematmul": True,
    "saveGt": False,
    "imag": 0,
    "dtype": "float32",
}  # real, unsaved Gt, nz=1

"""Seems to stop next tests from running
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1)])
def test_Gsize1(par):
    #Check error is raised if G has size 1 in any of the ranks
    with pytest.raises(NotImplementedError):
        _ = MPIFredholm1(
                np.ones((1,  par["nx"], par["ny"])),
                nz=par["nz"],
                saveGt=par["saveGt"],
                usematmul=par["usematmul"],
                dtype=par["dtype"],
            )
"""


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_Fredholm1(par):
    """Fredholm1 operator"""
    np.random.seed(42)

    _F = np.arange(par["nsl"] * par["nx"] * par["ny"]).reshape(
        par["nsl"], par["nx"], par["ny"]
    ).astype(par["dtype"])
    F = _F - par["imag"] * _F

    # split across ranks
    nsl_rank = local_split((par["nsl"], ), MPI.COMM_WORLD, Partition.SCATTER, 0)
    nsl_ranks = npp.concatenate(MPI.COMM_WORLD.allgather(nsl_rank))
    islin_rank = npp.insert(npp.cumsum(nsl_ranks)[:-1] , 0, 0)[rank]
    islend_rank = npp.cumsum(nsl_ranks)[rank]
    Frank = F[islin_rank:islend_rank]

    Fop_MPI = MPIFredholm1(
        Frank,
        nz=par["nz"],
        saveGt=par["saveGt"],
        usematmul=par["usematmul"],
        dtype=par["dtype"],
    )

    x = DistributedArray(global_shape=par["nsl"] * par["ny"] * par["nz"],
                         partition=pylops_mpi.Partition.BROADCAST,
                         dtype=par["dtype"], engine=backend)
    x[:] = 1. + par["imag"] * 1.
    x_global = x.asarray()
    # Forward
    y_dist = Fop_MPI @ x
    y = y_dist.asarray()
    # Adjoint
    y_adj_dist = Fop_MPI.H @ y_dist
    y_adj = y_adj_dist.asarray()
    # Dot test
    dottest(Fop_MPI, x, y_dist, par["nsl"] * par["nx"] * par["nz"], par["nsl"] * par["ny"] * par["nz"])

    if rank == 0:
        Fop = pylops.signalprocessing.Fredholm1(
            F,
            nz=par["nz"],
            saveGt=par["saveGt"],
            usematmul=False,
            dtype=par["dtype"],
        )

        assert Fop_MPI.shape == Fop.shape
        y_np = Fop @ x_global
        y_adj_np = Fop.H @ y_np
        assert_allclose(y, y_np, rtol=1e-14)
        assert_allclose(y_adj, y_adj_np, rtol=1e-14)


def _scatter_tile(tile, global_size, comm):
    vector = DistributedArray(
        global_shape=global_size,
        local_shapes=comm.allgather(tile.size),
        base_comm=comm,
        partition=Partition.SCATTER,
        dtype=tile.dtype,
        engine=backend,
    )
    vector[:] = tile.ravel()
    return vector


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("saveGt,kernel_dtype,input_dtype,nz", [
    (True, "float32", "float64", 3),
    (False, "complex64", "complex64", 1),
])
def test_Fredholm1SUMMA(saveGt, kernel_dtype, input_dtype, nz):
    """Check each rank's tiles against independent dense forward and adjoint products."""
    comm = MPI.COMM_WORLD
    # Use every rank: eight ranks exercise two separate 2 x 2 SUMMA grids.
    p = max(i for i in range(1, math.isqrt(size) + 1) if size % (i * i) == 0)
    pb = size // (p * p)
    batch, grid_rank = divmod(rank, p * p)
    row, col = divmod(grid_rank, p)
    nsl = pb + 1
    nx, ny = (5, 4) if saveGt else (4, 5)

    # Batch zero has two slices. Every other batch has one.
    slices = slice(0, 2) if batch == 0 else slice(batch + 1, batch + 2)

    def block(length, coord):
        width = (length + p - 1) // p
        return slice(coord * width, (coord + 1) * width)

    rng = npp.random.default_rng(42)

    def samples(shape, dtype):
        values = rng.standard_normal(shape)
        if npp.issubdtype(dtype, npp.complexfloating):
            values = values + 1j * rng.standard_normal(shape)
        return np.asarray(values, dtype=dtype)

    G = samples((nsl, nx, ny), kernel_dtype)
    model = samples((nsl, ny, nz), input_dtype)
    data = samples((nsl, nx, nz), input_dtype)
    kernel_tile = (slices, block(nx, row), block(ny, col))
    model_tile = (slices, block(ny, row), block(nz, col))
    data_tile = (slices, block(nx, row), block(nz, col))

    op = MPIFredholm1SUMMA(
        G[kernel_tile], nz=nz, dtype=kernel_dtype, base_comm=comm, saveGt=saveGt,
        pb=pb if saveGt else None, nsl_global=None if saveGt else nsl,
    )
    u = _scatter_tile(model[model_tile], model.size, comm)
    v = _scatter_tile(data[data_tile], data.size, comm)
    forward = op @ u
    adjoint = op.H @ v
    tol = 10 * np.finfo(input_dtype).eps
    assert dottest(op, u, v, rtol=tol, atol=tol)
    assert (op.pb, op.p) == (pb, p)
    assert forward.dtype == adjoint.dtype == np.result_type(kernel_dtype, input_dtype)
    assert_allclose(forward.local_array, (G @ model)[data_tile].ravel(), rtol=tol, atol=tol)
    expected = G.conj().transpose(0, 2, 1) @ data
    assert_allclose(adjoint.local_array, expected[model_tile].ravel(), rtol=tol, atol=tol)


@pytest.mark.mpi(min_size=2)
def test_Fredholm1SUMMA_inconsistent_kernel():
    # Only the last batch has the wrong x extent. All ranks must reject it.
    kernel = np.ones((1, 2 if rank == size - 1 else 1, 3))
    with pytest.raises(ValueError, match="batch and grid layout"):
        MPIFredholm1SUMMA(kernel, nz=2, pb=size)
