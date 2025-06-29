import numpy as np
from mpi4py import MPI
import math
import pylops_mpi

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M = 8 #512
N = 8 #512
K = 8 #512

A_shape  = (M,K)
B_shape  = (K,N)
C_shape  = (M,N)

p_prime = math.isqrt(size)
assert p_prime*p_prime == size, "Number of processes must be a perfect square"

# Create A with 2D block-cyclic structure
A_data = np.arange(int(A_shape[0]*A_shape[1])).reshape(A_shape)
A = A_data.reshape(p_prime, M//p_prime, p_prime, K//p_prime).transpose(1, 0, 2, 3).reshape(M//p_prime, -1)

# Create B with 2D block-cyclic structure
B_data = np.arange(int(B_shape[0]*B_shape[1])).reshape(B_shape)
B = B_data.reshape(p_prime, K//p_prime, p_prime, N//p_prime).transpose(1, 0, 2, 3).reshape(K//p_prime, -1)

A_dist = pylops_mpi.DistributedArray.to_dist(A,
                                          partition=pylops_mpi.Partition.SCATTER,
                                          axis=1)
B_dist = pylops_mpi.DistributedArray.to_dist(B,
                                          partition=pylops_mpi.Partition.SCATTER,
                                          axis=1)

C_dist = pylops_mpi.DistributedArray(global_shape=(M // p_prime, N * p_prime),
                                    partition=pylops_mpi.Partition.SCATTER,
                                     axis=1)
if rank == 0: print(A_dist.local_array)

i, j = divmod(rank, p_prime)
row_comm = comm.Split(color=i, key=j)
col_comm = comm.Split(color=j, key=i)

c_local = np.zeros((M//p_prime, N//p_prime))
for k in range(p_prime):
    Atemp=A_dist.local_array.copy() if j==k else np.empty_like(A_dist.local_array)
    Btemp=B_dist.local_array.copy() if i==k else np.empty_like(B_dist.local_array)
    rootA=i*p_prime+k; rootB=k*p_prime+j
    row_comm.Bcast([Atemp,MPI.FLOAT],root=k)
    col_comm.Bcast([Btemp,MPI.FLOAT],root=k)
    # print(f"[Rank {rank}] iter{k} after : received A from {rootA}, B from {rootB}, A0={Atemp.flat[0]},B0={Btemp.flat[0]}")
    c_local += Atemp @ Btemp

C_dist.local_array[:] = c_local
C_temp = C_dist.asarray().reshape((M,N))
C      = C_temp.reshape(M//p_prime, p_prime, p_prime, N//p_prime).transpose(1, 0, 2, 3).reshape(M, N)

if rank == 0 :
    print("expected:\n",A_data @ B_data)
    print("calculated:\n",C)