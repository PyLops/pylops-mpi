"""Test the MPIFredholm1 class
    Designed to run with n processes
    $ mpiexec -n 10 pytest test_fredholm.py --with-mpi
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
import numpy as npp
import math
from mpi4py import MPI
import pytest

import pylops
import pylops_mpi

from pylops_mpi import DistributedArray
from pylops_mpi.DistributedArray import local_split, Partition
from pylops_mpi.signalprocessing import MPIFredholm1
from pylops_mpi.signalprocessing.Fredholm1 import MPIFredholm1SUMMA
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

parsumma1 = {
    "nsl": 3,
    "ny": 5,
    "nx": 4,
    "nz": 3,
    "saveGt": True,
    "imag": 0,
    "dtype": "float32",
}
parsumma2 = {
    "nsl": 2,
    "ny": 4,
    "nx": 5,
    "nz": 3,
    "saveGt": False,
    "imag": 1j,
    "dtype": "complex64",
}


def _active_summa_comm(base_comm):
    size = base_comm.Get_size()
    p = math.isqrt(size)
    active_size = p * p
    if base_comm.Get_rank() >= active_size:
        return None, False
    if active_size == size:
        return base_comm, True
    active_ranks = list(range(active_size))
    group = base_comm.Get_group().Incl(active_ranks)
    comm = base_comm.Create_group(group)
    return comm, True


def _assemble_summa_tiles(op, local_tiles, nsl, nrows, ncols,
                          row_block, col_block, engine):
    comm = op.base_comm
    comm_nccl = op.base_comm_nccl if engine == "cupy" else None
    tiles = op._allgather(comm, comm_nccl, local_tiles, engine=engine)
    out = np.zeros((nsl, nrows, ncols), dtype=local_tiles.dtype)
    for rank_in_group in range(comm.Get_size()):
        row_id, col_id = divmod(rank_in_group, op.p)
        tile = tiles[rank_in_group]
        if tile.size == 0:
            continue
        rs = row_id * row_block
        cs = col_id * col_block
        rn = tile.shape[1]
        cn = tile.shape[2]
        out[:, rs:rs + rn, cs:cs + cn] = tile
    return out


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


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("par", [(parsumma1), (parsumma2)])
def test_Fredholm1SUMMA(par):
    """MPIFredholm1SUMMA operator"""
    np.random.seed(42)

    comm, is_active = _active_summa_comm(MPI.COMM_WORLD)
    if not is_active:
        return

    comm_rank = comm.Get_rank()
    p = math.isqrt(comm.Get_size())

    dtype = np.dtype(par["dtype"])
    if dtype == np.complex64 or dtype == np.float32:
        base_float_dtype = np.float32
    else:
        base_float_dtype = np.float64

    nsl = par["nsl"]
    nx = par["nx"]
    ny = par["ny"]
    nz = par["nz"]

    _G = np.arange(nsl * nx * ny,
                   dtype=base_float_dtype).reshape(nsl, nx, ny)
    G = (_G - par["imag"] * _G).astype(dtype)

    _M = np.arange(nsl * ny * nz,
                   dtype=base_float_dtype).reshape(nsl, ny, nz)
    M = (_M + par["imag"] * _M).astype(dtype)

    bn = (nx + p - 1) // p
    bk = (ny + p - 1) // p
    bm = (nz + p - 1) // p

    row_id, col_id = divmod(comm_rank, p)
    rs = row_id * bn
    re = min(rs + bn, nx)
    cs = col_id * bk
    ce = min(cs + bk, ny)
    ms = row_id * bk
    me = min(ms + bk, ny)
    zs = col_id * bm
    ze = min(zs + bm, nz)

    G_local = G[:, rs:re, cs:ce]
    M_local = M[:, ms:me, zs:ze]

    Fop_MPI = MPIFredholm1SUMMA(
        G_local,
        nz=nz,
        nsl_global=nsl,
        saveGt=par["saveGt"],
        pb=1,
        base_comm=comm,
        dtype=par["dtype"],
    )

    local_k = max(0, me - ms)
    local_n = max(0, re - rs)
    local_m = max(0, ze - zs)
    local_x_size = nsl * local_k * local_m
    local_shapes = comm.allgather(local_x_size)

    x_dist = DistributedArray(
        global_shape=nsl * ny * nz,
        local_shapes=local_shapes,
        partition=Partition.SCATTER,
        base_comm=comm,
        dtype=par["dtype"],
        engine=backend,
    )
    x_dist.local_array[:] = M_local.ravel()

    # Forward and adjoint
    y_dist = Fop_MPI @ x_dist
    xadj_dist = Fop_MPI.H @ y_dist

    # Dot test
    dottest(Fop_MPI, x_dist, y_dist,
            nsl * nx * nz, nsl * ny * nz)

    y_local = y_dist.local_array.reshape(nsl, local_n, local_m)
    y = _assemble_summa_tiles(
        Fop_MPI, y_local, nsl, nx, nz, bn, bm, backend
    )

    xadj_local = xadj_dist.local_array.reshape(nsl, local_k, local_m)
    xadj = _assemble_summa_tiles(
        Fop_MPI, xadj_local, nsl, ny, nz, bk, bm, backend
    )

    if comm_rank == 0:
        y_np = np.matmul(G, M)
        xadj_np = np.matmul(G.conj().transpose(0, 2, 1), y_np)
        rtol = np.finfo(base_float_dtype).resolution
        assert_allclose(
            y.squeeze(),
            y_np.squeeze(),
            rtol=rtol,
            err_msg=f"Rank {comm_rank}: Forward verification failed."
        )
        assert_allclose(
            xadj.squeeze(),
            xadj_np.squeeze(),
            rtol=rtol,
            err_msg=f"Rank {comm_rank}: Adjoint verification failed."
        )
