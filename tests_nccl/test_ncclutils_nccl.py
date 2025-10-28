"""Test basic NCCL functionalities in _nccl
Designed to run with n GPUs (with 1 MPI process per GPU)
$ mpiexec -n 10 pytest test_ncclutils_nccl.py --with-mpi
"""
from mpi4py import MPI
import numpy as np
import cupy as cp
from numpy.testing import assert_allclose
import pytest

from pylops_mpi.utils._nccl import initialize_nccl_comm, nccl_allgather
from pylops_mpi.utils._common import _prepare_allgather_inputs, _unroll_allgather_recv

np.random.seed(42)

nccl_comm = initialize_nccl_comm()

par1 = {'n': 3, 'dtype': np.float64}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), ])
def test_allgather_samesize(par):
    """Test nccl_allgather with arrays of same size"""
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    # Local array
    local_array = rank * cp.ones(par['n'], dtype=par['dtype'])

    # Gathered array
    gathered_array = nccl_allgather(nccl_comm, local_array)

    # Compare with global array created in rank0
    if rank == 0:
        global_array = np.ones(par['n'] * size, dtype=par['dtype'])
        for irank in range(size):
            global_array[irank * par["n"]: (irank + 1) * par["n"]] = irank

        assert_allclose(
            gathered_array.get(),
            global_array,
            rtol=1e-14,
        )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), ])
def test_allgather_samesize_withrecbuf(par):
    """Test nccl_allgather with arrays of same size and rec_buf"""
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    # Local array
    local_array = rank * cp.ones(par['n'], dtype=par['dtype'])

    # Gathered array
    gathered_array = cp.zeros(par['n'] * size, dtype=par['dtype'])
    gathered_array = nccl_allgather(nccl_comm, local_array, recv_buf=gathered_array)

    # Compare with global array created in rank0
    if rank == 0:
        global_array = np.ones(par['n'] * size, dtype=par['dtype'])
        for irank in range(size):
            global_array[irank * par["n"]: (irank + 1) * par["n"]] = irank

        assert_allclose(
            gathered_array.get(),
            global_array,
            rtol=1e-14,
        )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), ])
def test_allgather_differentsize_withrecbuf(par):
    """Test nccl_allgather with arrays of different size and rec_buf"""
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    # Local array
    n = par['n'] + (1 if rank == size - 1 else 0)
    local_array = rank * cp.ones(n, dtype=par['dtype'])

    # Gathered array
    send_shapes = MPI.COMM_WORLD.allgather(local_array.shape)
    send_buf, recv_buf = _prepare_allgather_inputs(local_array, send_shapes, engine="cupy")
    recv_buf = nccl_allgather(nccl_comm, send_buf, recv_buf)
    chunks = _unroll_allgather_recv(recv_buf, send_buf.shape, send_shapes)
    gathered_array = cp.concatenate(chunks)

    # Compare with global array created in rank0
    if rank == 0:
        global_array = np.ones(par['n'] * size + 1, dtype=par['dtype'])
        for irank in range(size - 1):
            global_array[irank * par["n"]: (irank + 1) * par["n"]] = irank
        global_array[(size - 1) * par["n"]:] = size - 1

        assert_allclose(
            gathered_array.get(),
            global_array,
            rtol=1e-14,
        )
