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

    blk_cols_BC = int(math.ceil(N / p_prime))
    col_start_B = my_layer * blk_cols_BC
    col_end_B = min(N, col_start_B + blk_cols_BC)
    local_col_B_len = max(0, col_end_B - col_start_B)


    A_glob_real = np.arange(M * K, dtype=base_float_dtype).reshape(M, K)
    A_glob_imag = np.arange(M * K, dtype=base_float_dtype).reshape(M, K) * 0.5
    A_glob = (A_glob_real + cmplx * A_glob_imag).astype(dtype)

    B_glob_real = np.arange(K * N, dtype=base_float_dtype).reshape(K, N)
    B_glob_imag = np.arange(K * N, dtype=base_float_dtype).reshape(K, N) * 0.7
    B_glob = (B_glob_real + cmplx * B_glob_imag).astype(dtype)

    A_p = A_glob[row_start_A:row_end_A,:]
    B_p = B_glob[:,col_start_B:col_end_B]

    # Create SUMMAMatrixMult operator
    Aop = MPIMatrixMult(A_p, N, base_comm=comm, dtype=dtype_str)

    # Create DistributedArray for input x (representing B flattened)
    all_local_col_len = comm.allgather(local_col_B_len)
    total_cols = np.sum(all_local_col_len)

    x_dist = DistributedArray(
        global_shape=(K * total_cols),
        local_shapes=[K * cl_b for cl_b in all_local_col_len],
        partition=Partition.SCATTER,
        base_comm=comm,
        mask=[i // p_prime for i in range(size)],
        dtype=dtype
    )

    x_dist.local_array[:] = B_p.ravel()

    # Forward operation: y = A @ B (distributed)
    y_dist = Aop @ x_dist

    # Adjoint operation: xadj = A.H @ y (distributed)
    xadj_dist = Aop.H @ y_dist

    y_loc = A_glob @ B_glob
    xadj_loc = A_glob.conj().T @ y_loc

    col_start_C_dist   = my_layer * blk_cols_BC
    col_end_C_dist     = min(N, col_start_C_dist + blk_cols_BC)
    my_own_cols_C_dist = max(0, col_end_C_dist - col_start_C_dist)
    expected_y_shape   = (M * my_own_cols_C_dist,)

    assert y_dist.local_array.shape == expected_y_shape, (
        f"Rank {rank}: y_dist shape {y_dist.local_array.shape} != expected {expected_y_shape}"
    )

    if y_dist.local_array.size > 0 and y_loc is not None and y_loc.size > 0:
        expected_y_slice = y_loc[:, col_start_C_dist:col_end_C_dist]
        assert_allclose(
            y_dist.local_array,
            expected_y_slice.ravel(),
            rtol=np.finfo(np.dtype(dtype)).resolution,
            err_msg=f"Rank {rank}: Forward verification failed."
        )

    # Verify adjoint operation (xadj = A.H @ y)
    expected_xadj_shape = (K * my_own_cols_C_dist,)
    assert xadj_dist.local_array.shape == expected_xadj_shape, (
        f"Rank {rank}: z_dist shape {xadj_dist.local_array.shape} != expected {expected_xadj_shape}"
    )

    # Verify adjoint result values
    if xadj_dist.local_array.size > 0 and xadj_loc  is not None and xadj_loc .size > 0:
        expected_xadj_slice = xadj_loc [:, col_start_C_dist:col_end_C_dist]
        assert_allclose(
            xadj_dist.local_array,
            expected_xadj_slice.ravel(),
            rtol=np.finfo(np.dtype(dtype)).resolution,
            err_msg=f"Rank {rank}: Adjoint verification failed."
        )

    group_comm.Free()
    layer_comm.Free()