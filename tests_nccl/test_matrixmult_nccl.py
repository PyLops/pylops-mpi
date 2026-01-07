"""Test the MPIMatrixMult class with NCCL
    Designed to run with n GPUs (with 1 MPI process per GPU)
    $ mpiexec -n 10 pytest test_matrixmult_nccl.py --with-mpi
"""
import math

import numpy as np
import cupy as cp
from numpy.testing import assert_allclose
from mpi4py import MPI
import pytest

from pylops.basicoperators import Conj, FirstDerivative
from pylops_mpi import DistributedArray, Partition
from pylops_mpi.basicoperators import MPIBlockDiag, MPIMatrixMult, \
    local_block_split, block_gather
from pylops_mpi.utils._nccl import initialize_nccl_comm

np.random.seed(42)

nccl_comm = initialize_nccl_comm()
base_comm = MPI.COMM_WORLD
size = base_comm.Get_size()
rank = base_comm.Get_rank()

# Define test cases: (N, K, M, dtype_str)
# M, K, N are matrix dimensions A(N,K), B(K,M)
# P_prime will be ceil(sqrt(size)).
test_params = [
    pytest.param(64, 64, 64, "float64", id="f32_64_64_64"),
    pytest.param(37, 37, 37, "float64", id="f32_37_37_37"),
    pytest.param(50, 30, 40, "float64", id="f64_50_30_40"),
    # temporarely removed as sometimes crashed CI... to be investigated
    # pytest.param(22, 20, 16, "complex64", id="c64_22_20_16"),
    pytest.param(3, 4, 5, "float32", id="f32_3_4_5"),
    pytest.param(1, 2, 1, "float64", id="f64_1_2_1",),
    pytest.param(2, 1, 3, "float32", id="f32_2_1_3",),
]


def _ensure_square_grid():
    p_prime = math.isqrt(size)
    if p_prime * p_prime != size:
        pytest.skip("MPIMatrixMult NCCL tests require a square number of ranks")
    return p_prime


def _reorganize_local_matrix(x_dist, nrows, ncols, blk_cols, p_prime):
    """Re-organize distributed array in local matrix"""
    x = x_dist.asarray(masked=True)
    col_counts = [min(blk_cols, ncols - j * blk_cols) for j in range(p_prime)]
    x_blocks = []
    offset = 0
    for cnt in col_counts:
        block_size = nrows * cnt
        x_block = x[offset: offset + block_size]
        if len(x_block) != 0:
            x_blocks.append(x_block.reshape(nrows, cnt))
        offset += block_size
    return cp.hstack(x_blocks)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("N, K, M, dtype_str", test_params)
def test_MPIMatrixMult_block_nccl(N, K, M, dtype_str):
    """MPIMatrixMult operator with kind=`block` and NCCL"""
    p_prime = _ensure_square_grid()
    if min(N, M) < p_prime:
        pytest.skip("MPIMatrixMult block test requires N and M >= sqrt(size)")

    dtype = np.dtype(dtype_str)
    cmplx = 1j if np.issubdtype(dtype, np.complexfloating) else 0
    base_float_dtype = np.float32 if dtype == np.complex64 else np.float64

    row_id, col_id = divmod(rank, p_prime)
    cols_id = base_comm.allgather(col_id)

    # Calculate local matrix dimensions
    blk_rows_A = int(math.ceil(N / p_prime))
    row_start_A = col_id * blk_rows_A
    row_end_A = min(N, row_start_A + blk_rows_A)

    blk_cols_X = int(math.ceil(M / p_prime))
    col_start_X = row_id * blk_cols_X
    col_end_X = min(M, col_start_X + blk_cols_X)
    local_col_X_len = max(0, col_end_X - col_start_X)

    # Fill local matrices
    A_glob_real = cp.arange(N * K, dtype=base_float_dtype).reshape(N, K)
    A_glob_imag = cp.arange(N * K, dtype=base_float_dtype).reshape(N, K) * 0.5
    A_glob = (A_glob_real + cmplx * A_glob_imag).astype(dtype)

    X_glob_real = cp.arange(K * M, dtype=base_float_dtype).reshape(K, M)
    X_glob_imag = cp.arange(K * M, dtype=base_float_dtype).reshape(K, M) * 0.7
    X_glob = (X_glob_real + cmplx * X_glob_imag).astype(dtype)

    A_p = A_glob[row_start_A:row_end_A, :]
    X_p = X_glob[:, col_start_X:col_end_X]

    # Create MPIMatrixMult operator
    Aop = MPIMatrixMult(A_p, M, base_comm=base_comm,
                        dtype=dtype_str, kind="block",
                        base_comm_nccl=nccl_comm)

    # Create DistributedArray for input x (representing B flattened)
    all_local_col_len = base_comm.allgather(local_col_X_len)
    total_cols = np.sum(all_local_col_len)

    x_dist = DistributedArray(
        global_shape=(K * total_cols),
        local_shapes=[(K * cl_b) for cl_b in all_local_col_len],
        partition=Partition.SCATTER,
        base_comm_nccl=nccl_comm,
        mask=[i % p_prime for i in range(size)],
        dtype=dtype,
        engine="cupy"
    )

    x_dist.local_array[:] = X_p.ravel()

    # Forward operation: y = A @ x (distributed)
    y_dist = Aop @ x_dist

    # Adjoint operation: xadj = A.H @ y (distributed)
    xadj_dist = Aop.H @ y_dist

    # Re-organize in local matrix
    y = _reorganize_local_matrix(y_dist, N, M, blk_cols_X, p_prime)
    xadj = _reorganize_local_matrix(xadj_dist, K, M, blk_cols_X, p_prime)

    if rank == 0:
        A_glob_np = A_glob.get()
        X_glob_np = X_glob.get()
        y_loc = A_glob_np @ X_glob_np
        assert_allclose(
            y.get().squeeze(),
            y_loc.squeeze(),
            rtol=np.finfo(np.dtype(dtype)).resolution,
            err_msg=f"Rank {rank}: Forward verification failed."
        )

        xadj_loc = A_glob_np.conj().T @ y_loc
        assert_allclose(
            xadj.get().squeeze(),
            xadj_loc.squeeze(),
            rtol=np.finfo(np.dtype(dtype)).resolution,
            err_msg=f"Rank {rank}: Adjoint verification failed."
        )

    # Chain with another operator
    Dop = FirstDerivative(dims=(N, col_end_X - col_start_X),
                          axis=0, dtype=dtype)
    DBop = MPIBlockDiag(ops=[Dop, ], base_comm=base_comm, mask=cols_id)
    Op = DBop @ Aop

    y1_dist = Op @ x_dist
    xadj1_dist = Op.H @ y1_dist

    # Re-organize in local matrix
    y1 = _reorganize_local_matrix(y1_dist, N, M, blk_cols_X, p_prime)
    xadj1 = _reorganize_local_matrix(xadj1_dist, K, M, blk_cols_X, p_prime)

    if rank == 0:
        A_glob_np = A_glob.get()
        X_glob_np = X_glob.get()
        Dop_glob = FirstDerivative(dims=(N, M), axis=0, dtype=dtype)
        y1_loc = (Dop_glob @ (A_glob_np @ X_glob_np).ravel()).reshape(N, M)
        assert_allclose(
            y1.get().squeeze(),
            y1_loc.squeeze(),
            rtol=np.finfo(np.dtype(dtype)).resolution,
            err_msg=f"Rank {rank}: Forward verification failed."
        )

        xadj1_loc = A_glob_np.conj().T @ (Dop_glob.H @ y1_loc.ravel()).reshape(N, M)
        assert_allclose(
            xadj1.get().squeeze(),
            xadj1_loc.squeeze(),
            rtol=np.finfo(np.dtype(dtype)).resolution,
            err_msg=f"Rank {rank}: Adjoint verification failed."
        )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("N, K, M, dtype_str", test_params)
def test_MPIMatrixMult_summa_nccl(N, K, M, dtype_str):
    """MPIMatrixMult operator with kind=`summa` and NCCL"""
    p_prime = _ensure_square_grid()
    if min(N, K, M) < p_prime:
        pytest.skip("MPIMatrixMult summa test requires N, K, M >= sqrt(size)")

    dtype = np.dtype(dtype_str)
    cmplx = 1j if np.issubdtype(dtype, np.complexfloating) else 0
    base_float_dtype = np.float32 if dtype == np.complex64 else np.float64

    # Fill local matrices
    A_glob_real = cp.arange(N * K, dtype=base_float_dtype).reshape(N, K)
    A_glob_imag = cp.arange(N * K, dtype=base_float_dtype).reshape(N, K) * 0.5
    A_glob = (A_glob_real + cmplx * A_glob_imag).astype(dtype)

    X_glob_real = cp.arange(K * M, dtype=base_float_dtype).reshape(K, M)
    X_glob_imag = cp.arange(K * M, dtype=base_float_dtype).reshape(K, M) * 0.7
    X_glob = (X_glob_real + cmplx * X_glob_imag).astype(dtype)

    A_slice = local_block_split((N, K), rank, base_comm)
    X_slice = local_block_split((K, M), rank, base_comm)

    A_p = A_glob[A_slice]
    X_p = X_glob[X_slice]

    # Create MPIMatrixMult operator
    Aop = MPIMatrixMult(A_p, M, base_comm=base_comm,
                        dtype=dtype_str, kind="summa",
                        base_comm_nccl=nccl_comm)

    x_dist = DistributedArray(
        global_shape=(K * M),
        local_shapes=base_comm.allgather(X_p.shape[0] * X_p.shape[1]),
        partition=Partition.SCATTER,
        base_comm_nccl=nccl_comm,
        dtype=dtype,
        engine="cupy",
    )

    x_dist.local_array[:] = X_p.ravel()

    # Forward operation: y = A @ x (distributed)
    y_dist = Aop @ x_dist

    # Adjoint operation: xadj = A.H @ y (distributed)
    xadj_dist = Aop.H @ y_dist

    # Re-organize in local matrix
    y = block_gather(y_dist, (N, M), base_comm)
    xadj = block_gather(xadj_dist, (K, M), base_comm)

    if rank == 0:
        A_glob_np = A_glob.get()
        X_glob_np = X_glob.get()
        y_loc = A_glob_np @ X_glob_np
        assert_allclose(
            y.get().squeeze(),
            y_loc.squeeze(),
            rtol=np.finfo(np.dtype(dtype)).resolution,
            err_msg=f"Rank {rank}: Forward verification failed."
        )

        xadj_loc = A_glob_np.conj().T @ y_loc
        assert_allclose(
            xadj.get().squeeze(),
            xadj_loc.squeeze(),
            rtol=np.finfo(np.dtype(dtype)).resolution,
            err_msg=f"Rank {rank}: Adjoint verification failed."
        )

    # Chain with another operator
    Dop = Conj(dims=(A_p.shape[0], X_p.shape[1]))
    DBop = MPIBlockDiag(ops=[Dop, ], base_comm=base_comm)
    Op = DBop @ Aop

    y1_dist = Op @ x_dist
    xadj1_dist = Op.H @ y1_dist

    # Re-organize in local matrix
    y1 = block_gather(y1_dist, (N, M), base_comm)
    xadj1 = block_gather(xadj1_dist, (K, M), base_comm)

    if rank == 0:
        A_glob_np = A_glob.get()
        X_glob_np = X_glob.get()
        y1_loc = ((A_glob_np @ X_glob_np).conj().ravel()).reshape(N, M)

        assert_allclose(
            y1.get().squeeze(),
            y1_loc.squeeze(),
            rtol=np.finfo(y1_loc.dtype).resolution,
            err_msg=f"Rank {rank}: Forward verification failed."
        )

        xadj1_loc = ((A_glob_np.conj().T @ y1_loc.conj()).ravel()).reshape(K, M)
        assert_allclose(
            xadj1.get().squeeze().ravel(),
            xadj1_loc.squeeze().ravel(),
            rtol=np.finfo(xadj1_loc.dtype).resolution,
            err_msg=f"Rank {rank}: Adjoint verification failed."
        )
