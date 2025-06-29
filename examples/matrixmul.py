from mpi4py import MPI
import math
import pylops_mpi
from pylops_mpi.basicoperators.MatrixMult import MPIMatrixMult
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 8
M = 8
K = 8

A_shape = (N, K)
B_shape = (K, M)
C_shape = (N, M)

p_prime = math.isqrt(size)
assert p_prime * p_prime == size, "Number of processes must be a perfect square"

A_data = np.arange(int(A_shape[0] * A_shape[1])).reshape(A_shape)
B_data = np.arange(int(B_shape[0] * B_shape[1])).reshape(B_shape)

N_starts, N_ends = MPIMatrixMult.block_distribute(N, p_prime)
M_starts, M_ends = MPIMatrixMult.block_distribute(M, p_prime)
K_starts, K_ends = MPIMatrixMult.block_distribute(K, p_prime)

i, j = divmod(rank, p_prime)
A_local = A_data[N_starts[i]:N_ends[i], K_starts[j]:K_ends[j]]
B_local = B_data[K_starts[i]:K_ends[i], M_starts[j]:M_ends[j]]

B_dist = pylops_mpi.DistributedArray(global_shape=(K*M),
                                     local_shapes=comm.allgather(B_local.shape[0] * B_local.shape[1]),
                                     base_comm=comm,
                                     partition=pylops_mpi.Partition.SCATTER)
B_dist.local_array[:] = B_local.flatten()

print(rank, A_local.shape)
Aop = MPIMatrixMult(A_local, M, base_comm=comm)
C_dist = Aop @ B_dist
C_temp = C_dist.asarray().reshape((N, M))
C      = C_temp.reshape(N // p_prime, p_prime, p_prime, M // p_prime).transpose(1, 0, 2, 3).reshape(N, M)

if rank == 0 :
    # print("expected:\n",np.allclose(A_data @ B_data, C))
    print("expected:\n", A_data @ B_data)
    print("calculated:\n",C)