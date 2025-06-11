import pytest
import numpy as np
from numpy.testing import assert_allclose
from mpi4py import MPI
import math

from pylops_mpi import DistributedArray, Partition
from pylops_mpi.basicoperators.MatrixMult import MPISUMMAMatrixMult

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
def test_MPIMatrixMult(M, K, N, dtype_str):
    dtype = np.dtype(dtype_str)

    cmplx = 1j if np.issubdtype(dtype, np.complexfloating) else 0
    base_float_dtype = np.float32 if dtype == np.complex64 else np.float64

    P_prime = int(math.ceil(math.sqrt(size)))
    C = int(math.ceil(size / P_prime))
    assert P_prime * C >= size  # Ensure process grid covers all processes

    my_group = rank % P_prime
    my_layer = rank // P_prime

    # Create sub-communicators
    layer_comm = comm.Split(color=my_layer, key=my_group)
    group_comm = comm.Split(color=my_group, key=my_layer)

    # Calculate local matrix dimensions
    blk_rows_A = int(math.ceil(M / P_prime))
    row_start_A = my_group * blk_rows_A
    row_end_A = min(M, row_start_A + blk_rows_A)
    my_own_rows_A = max(0, row_end_A - row_start_A)

    blk_cols_BC = int(math.ceil(N / P_prime))
    col_start_B = my_group * blk_cols_BC
    col_end_B = min(N, col_start_B + blk_cols_BC)
    my_own_cols_B = max(0, col_end_B - col_start_B)

    # Initialize local matrices
    A_p = np.empty((my_own_rows_A, K), dtype=dtype)
    B_p = np.empty((K, my_own_cols_B), dtype=dtype)

    # Generate and distribute test matrices
    A_glob, B_glob = None, None
    if rank == 0:
        # Create global matrices with complex components if needed
        A_glob_real = np.arange(M * K, dtype=base_float_dtype).reshape(M, K)
        A_glob_imag = np.arange(M * K, dtype=base_float_dtype).reshape(M, K) * 0.5
        A_glob = (A_glob_real + cmplx * A_glob_imag).astype(dtype)

        B_glob_real = np.arange(K * N, dtype=base_float_dtype).reshape(K, N)
        B_glob_imag = np.arange(K * N, dtype=base_float_dtype).reshape(K, N) * 0.7
        B_glob = (B_glob_real + cmplx * B_glob_imag).astype(dtype)

        # Distribute matrix blocks to all ranks
        for dest_rank in range(size):
            dest_my_group = dest_rank % P_prime

            # Calculate destination rank's block dimensions
            dest_row_start_A = dest_my_group * blk_rows_A
            dest_row_end_A = min(M, dest_row_start_A + blk_rows_A)
            dest_my_own_rows_A = max(0, dest_row_end_A - dest_row_start_A)

            dest_col_start_B = dest_my_group * blk_cols_BC
            dest_col_end_B = min(N, dest_col_start_B + blk_cols_BC)
            dest_my_own_cols_B = max(0, dest_col_end_B - dest_col_start_B)

            A_block_send = A_glob[dest_row_start_A:dest_row_end_A, :].copy()
            B_block_send = B_glob[:, dest_col_start_B:dest_col_end_B].copy()

            # Validate block shapes
            assert A_block_send.shape == (dest_my_own_rows_A, K)
            assert B_block_send.shape == (K, dest_my_own_cols_B)

            if dest_rank == 0:
                A_p, B_p = A_block_send, B_block_send
            else:
                if A_block_send.size > 0:
                    comm.Send(A_block_send, dest=dest_rank, tag=100 + dest_rank)
                if B_block_send.size > 0:
                    comm.Send(B_block_send, dest=dest_rank, tag=200 + dest_rank)
    else:
        if A_p.size > 0:
            comm.Recv(A_p, source=0, tag=100 + rank)
        if B_p.size > 0:
            comm.Recv(B_p, source=0, tag=200 + rank)

    comm.Barrier()

    # Create SUMMAMatrixMult operator
    Aop = MPISUMMAMatrixMult(A_p, N, base_comm=comm, dtype=dtype_str)

    # Create DistributedArray for input x (representing B flattened)
    all_my_own_cols_B = comm.allgather(my_own_cols_B)
    total_cols = sum(all_my_own_cols_B)
    local_shapes_x = [(K * cl_b,) for cl_b in all_my_own_cols_B]

    x_dist = DistributedArray(
        global_shape=(K * total_cols),
        local_shapes=local_shapes_x,
        partition=Partition.SCATTER,
        base_comm=comm,
        dtype=dtype
    )

    if B_p.size > 0:
        x_dist.local_array[:] = B_p.ravel()
    else:
        assert x_dist.local_array.size == 0, (
            f"Rank {rank}: B_p empty but x_dist.local_array not empty "
            f"(size {x_dist.local_array.size})"
        )

    # Forward operation: y = A @ B (distributed)
    y_dist = Aop @ x_dist
    y = y_dist.asarray(),

    # Adjoint operation: z = A.H @ y (distributed y representing C)
    y_adj_dist = Aop.H @ y_dist
    y_adj = y_adj_dist.asarray()

    if rank == 0:
        y_np = A_glob @ B_glob
        y_adj_np = A_glob.conj().T @ y_np
        assert_allclose(
            y,
            y_np.ravel(),
            rtol=1e-14,
            err_msg=f"Rank {rank}: Forward verification failed."
        )

        assert_allclose(
            y_adj,
            y_adj_np.ravel(),
            rtol=1e-14,
            err_msg=f"Rank {rank}: Adjoint verification failed."
        )

    group_comm.Free()
    layer_comm.Free()