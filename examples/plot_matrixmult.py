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
from pylops_mpi.basicoperators.MatrixMult import MPIMatrixMult

np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()

P_prime = int(math.ceil(math.sqrt(n_procs)))
C = int(math.ceil(n_procs / P_prime))

if (P_prime * C) != n_procs:
    print("No. of procs has to be a square number")
    exit(-1)

# matrix dims
M = 33
K = 34
N = 37

A = np.random.rand(M * K).astype(dtype=np.float32).reshape(M, K)
B = np.random.rand(K * N).astype(dtype=np.float32).reshape(K, N)

my_group = rank % P_prime
my_layer = rank // P_prime

# sub‐communicators
layer_comm = comm.Split(color=my_layer, key=my_group)  # all procs in same layer
group_comm = comm.Split(color=my_group, key=my_layer)  # all procs in same group


#Each rank will end up with:
#      - :math:`A_{p} \in \mathbb{R}^{\text{my\_own\_rows}\times K}`
#      - :math:`B_{p} \in \mathbb{R}^{K\times \text{my\_own\_cols}}`
#    where
blk_rows = int(math.ceil(M / P_prime))
blk_cols = int(math.ceil(N / P_prime))

rs = my_group * blk_rows
re = min(M, rs + blk_rows)
my_own_rows = re - rs

cs = my_layer * blk_cols
ce = min(N, cs + blk_cols)
my_own_cols = ce - cs

A_p, B_p = A[rs:re, :].copy(), B[:, cs:ce].copy()

Aop = MPIMatrixMult(A_p, N, dtype="float32")
col_lens = comm.allgather(my_own_cols)
total_cols =  np.sum(col_lens)
x = DistributedArray(global_shape=K * total_cols,
                     local_shapes=[K * col_len for col_len in col_lens],
                     partition=Partition.SCATTER,
                     mask=[i % P_prime for i in range(comm.Get_size())],
                     base_comm=comm,
                     dtype="float32")
x[:] = B_p.flatten()
y = Aop @ x

# ======================= VERIFICATION =================-=============
y_loc = A @ B
xadj_loc = (A.T.dot(y_loc.conj())).conj()


expected_y_loc = y_loc[:, cs:ce].flatten().astype(np.float32)
expected_xadj_loc = xadj_loc[:, cs:ce].flatten().astype(np.float32)

xadj = Aop.H @ y
if not np.allclose(y.local_array, expected_y_loc, rtol=1e-6):
    print(f"RANK {rank}: FORWARD VERIFICATION FAILED")
    print(f'{rank} local: {y.local_array}, expected: {y_loc[:, cs:ce]}')
else:
    print(f"RANK {rank}: FORWARD VERIFICATION PASSED")

if not np.allclose(xadj.local_array, expected_xadj_loc, rtol=1e-6):
    print(f"RANK {rank}: ADJOINT VERIFICATION FAILED")
    print(f'{rank} local: {xadj.local_array}, expected: {xadj_loc[:, cs:ce]}')
else:
    print(f"RANK {rank}: ADJOINT VERIFICATION PASSED")

