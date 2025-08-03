"""Test the MPIMatrixMult class
    Designed to run with n processes
    $ mpiexec -n 10 pytest test_matrixmult.py --with-mpi
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

from pylops.basicoperators import FirstDerivative, Identity
from pylops_mpi import DistributedArray, Partition
from pylops_mpi.basicoperators import MPIMatrixMult, MPIBlockDiag

np.random.seed(42)
base_comm = MPI.COMM_WORLD
size = base_comm.Get_size()

# Define test cases: (N, K, M, dtype_str)
# M, K, N are matrix dimensions A(N,K), B(K,M)
# P_prime will be ceil(sqrt(size)).
test_params = [
    pytest.param(37, 37, 37, "float64", id="f32_37_37_37"),
    pytest.param(50, 30, 40, "float64", id="f64_50_30_40"),
    pytest.param(22, 20, 16, "complex64", id="c64_22_20_16"),
    pytest.param(3, 4, 5, "float32", id="f32_3_4_5"),
    pytest.param(1, 2, 1, "float64", id="f64_1_2_1",),
    pytest.param(2, 1, 3, "float32", id="f32_2_1_3",),
]

def _reorganize_local_matrix(x_dist, N, M, blk_cols, p_prime):
    """Re-organize distributed array in local matrix
    """
    x = x_dist.asarray(masked=True)
    col_counts = [min(blk_cols, M - j * blk_cols) for j in range(p_prime)]
    x_blocks = []
    offset = 0
    for cnt in col_counts:
        block_size = N * cnt
        x_block = x[offset: offset + block_size]
        if len(x_block) != 0:
            x_blocks.append(
                x_block.reshape(N, cnt)
            )
        offset += block_size
    x = np.hstack(x_blocks)
    return x


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("N, K, M, dtype_str", test_params)
def test_MPIMatrixMult(N, K, M, dtype_str):
    dtype = np.dtype(dtype_str)

    cmplx = 1j if np.issubdtype(dtype, np.complexfloating) else 0
    base_float_dtype = np.float32 if dtype == np.complex64 else np.float64

    comm, rank, row_id, col_id, is_active = \
        MPIMatrixMult.active_grid_comm(base_comm, N, M)
    if not is_active: return

    size = comm.Get_size()
    p_prime = math.isqrt(size)
    cols_id = comm.allgather(col_id)

    # Calculate local matrix dimensions
    blk_rows_A = int(math.ceil(N / p_prime))
    row_start_A = col_id * blk_rows_A
    row_end_A = min(N, row_start_A + blk_rows_A)

    blk_cols_X = int(math.ceil(M / p_prime))
    col_start_X = row_id * blk_cols_X
    col_end_X = min(M, col_start_X + blk_cols_X)
    local_col_X_len = max(0, col_end_X - col_start_X)

    # Fill local matrices
    A_glob_real = np.arange(N * K, dtype=base_float_dtype).reshape(N, K)
    A_glob_imag = np.arange(N * K, dtype=base_float_dtype).reshape(N, K) * 0.5
    A_glob = (A_glob_real + cmplx * A_glob_imag).astype(dtype)

    X_glob_real = np.arange(K * M, dtype=base_float_dtype).reshape(K, M)
    X_glob_imag = np.arange(K * M, dtype=base_float_dtype).reshape(K, M) * 0.7
    X_glob = (X_glob_real + cmplx * X_glob_imag).astype(dtype)

    A_p = A_glob[row_start_A:row_end_A, :]
    X_p = X_glob[:, col_start_X:col_end_X]

    # Create MPIMatrixMult operator
    Aop = MPIMatrixMult(A_p, M, base_comm=comm, dtype=dtype_str)

    # Create DistributedArray for input x (representing B flattened)
    all_local_col_len = comm.allgather(local_col_X_len)
    total_cols = npp.sum(all_local_col_len)

    x_dist = DistributedArray(
        global_shape=(K * total_cols),
        local_shapes=[K * cl_b for cl_b in all_local_col_len],
        partition=Partition.SCATTER,
        base_comm=comm,
        mask=[i % p_prime for i in range(size)],
        dtype=dtype,
        engine=backend
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
        y_loc = A_glob @ X_glob
        assert_allclose(
            y.squeeze(),
            y_loc.squeeze(),
            rtol=np.finfo(np.dtype(dtype)).resolution,
            err_msg=f"Rank {rank}: Forward verification failed."
        )

        xadj_loc = A_glob.conj().T @ y_loc
        assert_allclose(
            xadj.squeeze(),
            xadj_loc.squeeze(),
            rtol=np.finfo(np.dtype(dtype)).resolution,
            err_msg=f"Rank {rank}: Adjoint verification failed."
        )

    # Chain with another operator
    Dop = FirstDerivative(dims=(N, col_end_X - col_start_X),
                          axis=0, dtype=dtype)
    DBop = MPIBlockDiag(ops=[Dop, ], base_comm=comm, mask=cols_id)
    Op = DBop @ Aop

    y1_dist = Op @ x_dist
    xadj1_dist = Op.H @ y1_dist

    # Re-organize in local matrix
    y1 = _reorganize_local_matrix(y1_dist, N, M, blk_cols_X, p_prime)
    xadj1 = _reorganize_local_matrix(xadj1_dist, K, M, blk_cols_X, p_prime)

    if rank == 0:
        Dop_glob = FirstDerivative(dims=(N, M), axis=0, dtype=dtype)
        y1_loc = (Dop_glob @ (A_glob @ X_glob).ravel()).reshape(N, M)
        assert_allclose(
            y1.squeeze(),
            y1_loc.squeeze(),
            rtol=np.finfo(np.dtype(dtype)).resolution,
            err_msg=f"Rank {rank}: Forward verification failed."
        )

        xadj1_loc = A_glob.conj().T @ (Dop_glob.H @ y1_loc.ravel()).reshape(N, M)
        assert_allclose(
            xadj1.squeeze(),
            xadj1_loc.squeeze(),
            rtol=np.finfo(np.dtype(dtype)).resolution,
            err_msg=f"Rank {rank}: Adjoint verification failed."
        )
