"""
Distributed Matrix Multiplication
=========================
This example shows how to use the :py:class:`pylops_mpi.basicoperators.MatrixMultiply.SUMMAMatrixMult`.
This class provides a way to distribute arrays across multiple processes in
a parallel computing environment.
"""

import math
import numpy as np
from mpi4py import MPI

from pylops_mpi import DistributedArray, Partition
from pylops_mpi.basicoperators.MatrixMult import MPISUMMAMatrixMult

np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()

P_prime = int(math.ceil(math.sqrt(n_procs)))
C = int(math.ceil(n_procs / P_prime))

if P_prime * C < n_procs:
    print("No. of procs has to be a square number")
    exit(-1)

# matrix dims
M = 32
K = 32
N = 35

blk_rows = int(math.ceil(M / P_prime))
blk_cols = int(math.ceil(N / P_prime))

my_group = rank % P_prime
my_layer = rank // P_prime

# sub‐communicators
layer_comm = comm.Split(color=my_layer, key=my_group)  # all procs in same layer
group_comm = comm.Split(color=my_group, key=my_layer)  # all procs in same group

# Each rank will end up with:
#   A_p: shape (my_own_rows, K)
#   B_p: shape (K, my_own_cols)
# where
row_start = my_group * blk_rows
row_end = min(M, row_start + blk_rows)
my_own_rows = row_end - row_start

col_start = my_group * blk_cols  # note: same my_group index on cols
col_end = min(N, col_start + blk_cols)
my_own_cols = col_end - col_start

# ======================= BROADCASTING THE SLICES =======================
if rank == 0:
    A = np.arange(M * K, dtype=np.float32).reshape(M, K)
    B = np.arange(K * N, dtype=np.float32).reshape(K, N)
    for dest in range(n_procs):
        pg = dest % P_prime
        rs = pg * blk_rows;
        re = min(M, rs + blk_rows)
        cs = pg * blk_cols;
        ce = min(N, cs + blk_cols)
        a_block, b_block = A[rs:re, :], B[:, cs:ce]
        if dest == 0:
            A_p, B_p = a_block, b_block
        else:
            comm.Send(a_block, dest=dest, tag=100 + dest)
            comm.Send(b_block, dest=dest, tag=200 + dest)
else:
    A_p = np.empty((my_own_rows, K), dtype=np.float32)
    B_p = np.empty((K, my_own_cols), dtype=np.float32)
    comm.Recv(A_p, source=0, tag=100 + rank)
    comm.Recv(B_p, source=0, tag=200 + rank)

comm.Barrier()

Aop = MPISUMMAMatrixMult(A_p, N)
col_lens = comm.allgather(my_own_cols)
total_cols =  np.sum(col_lens)
x = DistributedArray(global_shape=K * total_cols,
                     local_shapes=[K * col_len for col_len in col_lens],
                     partition=Partition.SCATTER,
                     mask=[i % P_prime for i in range(comm.Get_size())],
                     dtype=np.float32)
x[:] = B_p.flatten()
y = Aop @ x

# ======================= VERIFICATION =================-=============
A = np.arange(M * K).reshape(M, K).astype(np.float32)
B = np.arange(K * N).reshape(K, N).astype(np.float32)
C_true = A @ B
Z_true = (A.T.dot(C_true.conj())).conj()

col_start = my_layer * blk_cols  # note: same my_group index on cols
col_end = min(N, col_start + blk_cols)
my_own_cols = col_end - col_start
expected_y = C_true[:, col_start:col_end].flatten()

xadj = Aop.H @ y

if not np.allclose(y.local_array, expected_y, atol=1e-6, rtol=1e-14):
    print(f"RANK {rank}: FORWARD VERIFICATION FAILED")
    print(f'{rank} local: {y.local_array}, expected: {C_true[:, col_start:col_end]}')
else:
    print(f"RANK {rank}: FORWARD VERIFICATION PASSED")

expected_z = Z_true[:, col_start:col_end].flatten()
if not np.allclose(xadj.local_array, expected_z, atol=1e-6, rtol=1e-14):
    print(f"RANK {rank}: ADJOINT VERIFICATION FAILED")
    print(f'{rank} local: {xadj.local_array}, expected: {Z_true[:, col_start:col_end]}')
else:
    print(f"RANK {rank}: ADJOINT VERIFICATION PASSED")
