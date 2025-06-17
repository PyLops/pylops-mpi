import pytest
import numpy as np
from numpy.testing import assert_allclose
from mpi4py import MPI
import math
import sys

from pylops_mpi import DistributedArray, Partition
from pylops_mpi.basicoperators.MatrixMult import MPIMatrixMult

np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define test cases: (N K, M, dtype_str)
# M, K, N are matrix dimensions A(N,K), B(K,M)
# P_prime will be ceil(sqrt(size)).
test_params = [
    pytest.param(37, 37, 37, "float32",   id="f32_37_37_37"),
    pytest.param(50, 30, 40, "float64",   id="f64_50_30_40"),
    pytest.param(22, 20, 16, "complex64", id="c64_22_20_16"),
    pytest.param( 3,  4,  5, "float32",   id="f32_3_4_5"),
    pytest.param( 1,  2,  1, "float64",   id="f64_1_2_1",),
    pytest.param( 2,  1,  3, "float32",   id="f32_2_1_3",),
]


@pytest.mark.mpi(min_size=1)  # SUMMA should also work for 1 process.
@pytest.mark.parametrize("M, K, N, dtype_str", test_params)
def test_SUMMAMatrixMult(N, K, M, dtype_str):
    p_prime = math.isqrt(size)
    C = p_prime
    if  p_prime * C != size:
        pytest.skip(f"Number of processes must be a square number, provided {size} instead...")

    dtype = np.dtype(dtype_str)

    cmplx = 1j if np.issubdtype(dtype, np.complexfloating) else 0
    base_float_dtype = np.float32 if dtype == np.complex64 else np.float64

    my_group = rank % p_prime
    my_layer = rank // p_prime

    # Create sub-communicators
    layer_comm = comm.Split(color=my_layer, key=my_group)
    group_comm = comm.Split(color=my_group, key=my_layer)

    # Calculate local matrix dimensions
    blk_rows_A = int(math.ceil(N / p_prime))
    row_start_A = my_group * blk_rows_A
    row_end_A = min(N, row_start_A + blk_rows_A)

    blk_cols_X = int(math.ceil(M / p_prime))
    col_start_X = my_layer * blk_cols_X
    col_end_X = min(M, col_start_X + blk_cols_X)
    local_col_X_len = max(0, col_end_X - col_start_X)

    A_glob_real = np.arange(N * K, dtype=base_float_dtype).reshape(N, K)
    A_glob_imag = np.arange(N * K, dtype=base_float_dtype).reshape(N, K) * 0.5
    A_glob = (A_glob_real + cmplx * A_glob_imag).astype(dtype)

    X_glob_real = np.arange(K * M, dtype=base_float_dtype).reshape(K, M)
    X_glob_imag = np.arange(K * M, dtype=base_float_dtype).reshape(K, M) * 0.7
    X_glob = (X_glob_real + cmplx * X_glob_imag).astype(dtype)

    A_p = A_glob[row_start_A:row_end_A,:]
    X_p = X_glob[:,col_start_X:col_end_X]

    # Create MPIMatrixMult operator
    Aop = MPIMatrixMult(A_p, M, base_comm=comm, dtype=dtype_str)

    # Create DistributedArray for input x (representing B flattened)
    all_local_col_len = comm.allgather(local_col_X_len)
    total_cols = np.sum(all_local_col_len)

    x_dist = DistributedArray(
        global_shape=(K * total_cols),
        local_shapes=[K * cl_b for cl_b in all_local_col_len],
        partition=Partition.SCATTER,
        base_comm=comm,
        mask=[i % p_prime for i in range(size)],
        dtype=dtype
    )

    x_dist.local_array[:] = X_p.ravel()

    # Forward operation: y = A @ B (distributed)
    y_dist = Aop @ x_dist
    # Adjoint operation: xadj = A.H @ y (distributed)
    xadj_dist = Aop.H @ y_dist

    y = y_dist.asarray(masked=True)
    col_counts = [min(blk_cols_X, M - j * blk_cols_X) for j in range(p_prime)]
    y_blocks = []
    offset = 0
    for cnt in col_counts:
        block_size = N * cnt
        y_blocks.append(
            y[offset: offset + block_size].reshape(N, cnt)
        )
        offset += block_size
    y = np.hstack(y_blocks)

    xadj = xadj_dist.asarray(masked=True)
    xadj_blocks = []
    offset = 0
    for cnt in col_counts:
        block_size = K * cnt
        xadj_blocks.append(
            xadj[offset: offset + block_size].reshape(K, cnt)
        )
        offset += block_size
    xadj = np.hstack(xadj_blocks)

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
            err_msg=f"Rank {rank}: Ajoint verification failed."
        )

    group_comm.Free()
    layer_comm.Free()