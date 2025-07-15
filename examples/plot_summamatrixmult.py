import math
import numpy as np
from mpi4py import MPI

import pylops_mpi
from pylops_mpi.basicoperators.MatrixMult import (local_block_spit,
                                                     block_gather,
                                                     MPISummaMatrixMult)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 9
M = 9
K = 9

A_shape = (N, K)
B_shape = (K, M)
C_shape = (N, M)

p_prime = math.isqrt(size)
assert p_prime * p_prime == size, "Number of processes must be a perfect square"

A_data = np.arange(int(A_shape[0] * A_shape[1])).reshape(A_shape)
B_data = np.arange(int(B_shape[0] * B_shape[1])).reshape(B_shape)

A_slice = local_block_spit(A_shape, rank, comm)
B_slice = local_block_spit(B_shape, rank, comm)
A_local = A_data[A_slice]
B_local = B_data[B_slice]
# A_local, (N_new, K_new) = block_distribute(A_data,rank, comm)
# B_local, (K_new, M_new) = block_distribute(B_data,rank, comm)

B_dist = pylops_mpi.DistributedArray(global_shape=(K * M),
                                     local_shapes=comm.allgather(B_local.shape[0] * B_local.shape[1]),
                                     base_comm=comm,
                                     partition=pylops_mpi.Partition.SCATTER)
B_dist.local_array[:] = B_local.flatten()

Aop = MPISummaMatrixMult(A_local, M, base_comm=comm)
C_dist = Aop @ B_dist
Z_dist = Aop.H @ C_dist

C = block_gather(C_dist, (N,M), (N,M), comm)
Z = block_gather(Z_dist, (K,M), (K,M), comm)
if rank == 0 :
    C_correct = np.allclose(A_data @ B_data, C)
    print("C expected: ", C_correct)
    if not C_correct:
        print("expected:\n", A_data @ B_data)
        print("calculated:\n",C)

    Z_correct = np.allclose((A_data.T.dot((A_data @ B_data).conj())).conj(), Z.astype(np.int32))
    print("Z expected: ", Z_correct)
    if not Z_correct:
        print("expected:\n", (A_data.T.dot((A_data @ B_data).conj())).conj())
        print("calculated:\n", Z.astype(np.int32))
