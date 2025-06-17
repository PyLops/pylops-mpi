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

# Define test cases: (M, K, N, dtype_str)
# M, K, N are matrix dimensions A(M,K), B(K,N)
# P_prime will be ceil(sqrt(size)).
test_params = [
    pytest.param(37, 37, 37, "float32",   id="f32_37_37_37"),
    pytest.param(40, 30, 50, "float64",   id="f64_40_30_50"),
    pytest.param(16, 20, 22, "complex64", id="c64_16_20_22"),
    pytest.param(5,   4,  3, "float32",   id="f32_5_4_3"),
    pytest.param(1,   2,  1, "float64",   id="f64_1_2_1",),
    pytest.param(3,   1,  2, "float32",   id="f32_3_1_2",),
]


@pytest.mark.mpi(min_size=1)  # SUMMA should also work for 1 process.
@pytest.mark.parametrize("M, K, N, dtype_str", test_params)
def test_SUMMAMatrixMult(M, K, N, dtype_str):
    dtype = np.dtype(dtype_str)

    cmplx = 1j if np.issubdtype(dtype, np.complexfloating) else 0
    base_float_dtype = np.float32 if dtype == np.complex64 else np.float64

    p_prime = int(math.ceil(math.sqrt(size)))
    C = int(math.ceil(size / p_prime))
    assert p_prime * C == size

    my_group = rank % p_prime
    my_layer = rank // p_prime

    # Create sub-communicators
    layer_comm = comm.Split(color=my_layer, key=my_group)
    group_comm = comm.Split(color=my_group, key=my_layer)

    # Calculate local matrix dimensions
    blk_rows_A = int(math.ceil(M / p_prime))
    row_start_A = my_group * blk_rows_A
    row_end_A = min(M, row_start_A + blk_rows_A)

    blk_cols_X = int(math.ceil(N / p_prime))
    col_start_X = my_layer * blk_cols_X
    col_end_X = min(N, col_start_X + blk_cols_X)
    local_col_X_len = max(0, col_end_X - col_start_X)

    A_glob_real = np.arange(M * K, dtype=base_float_dtype).reshape(M, K)
    A_glob_imag = np.arange(M * K, dtype=base_float_dtype).reshape(M, K) * 0.5
    A_glob = (A_glob_real + cmplx * A_glob_imag).astype(dtype)

    X_glob_real = np.arange(K * N, dtype=base_float_dtype).reshape(K, N)
    X_glob_imag = np.arange(K * N, dtype=base_float_dtype).reshape(K, N) * 0.7
    X_glob = (X_glob_real + cmplx * X_glob_imag).astype(dtype)

    A_p = A_glob[row_start_A:row_end_A,:]
    X_p = X_glob[:,col_start_X:col_end_X]

    # Create MPIMatrixMult operator
    Aop = MPIMatrixMult(A_p, N, base_comm=comm, dtype=dtype_str)

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

    y    = y_dist.asarray(masked=True)
    y    = y.reshape(p_prime, M, blk_cols_X)

    xadj = xadj_dist.asarray(masked=True)
    xadj = xadj.reshape(p_prime, K, blk_cols_X)

    if rank == 0:
        y_loc = (A_glob @ X_glob).squeeze()
        assert_allclose(
            y,
            y_loc,
            rtol=np.finfo(np.dtype(dtype)).resolution,
            err_msg=f"Rank {rank}: Forward verification failed."
        )

        xadj_loc = (A_glob.conj().T @ y_loc.conj()).conj().squeeze()
        assert_allclose(
            xadj,
            xadj_loc,
            rtol=np.finfo(np.dtype(dtype)).resolution,
            err_msg=f"Rank {rank}: Ajoint verification failed."
        )

    group_comm.Free()
    layer_comm.Free()