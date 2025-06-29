import numpy as np
from mpi4py import MPI
import math
import pylops_mpi

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M = 16 #512
N = 16 #512
K = 16 #512

A_shape  = (M,K)
B_shape  = (K,N)
C_shape  = (M,N)

p_prime = math.isqrt(size)
assert p_prime*p_prime == size
A = np.arange(int(A_shape[0]*A_shape[1])).reshape(A_shape).reshape((M//p_prime,-1))
B = np.arange(int(B_shape[0]*B_shape[1])).reshape(B_shape).reshape((K//p_prime,-1))

A_dist = pylops_mpi.DistributedArray.to_dist(A,
                                          partition=pylops_mpi.Partition.SCATTER,
                                          axis=1)
B_dist = pylops_mpi.DistributedArray.to_dist(B,
                                          partition=pylops_mpi.Partition.SCATTER,
                                          axis=1)

C_dist = pylops_mpi.DistributedArray(global_shape=(M // p_prime, N * p_prime),
                                    partition=pylops_mpi.Partition.SCATTER,
                                     axis=1)



p    = int(np.sqrt(size))
i, j = divmod(rank, p)
row_comm = comm.Split(color=i, key=j)
col_comm = comm.Split(color=j, key=i)

c = np.zeros((M//p, N//p), dtype=np.float32)
for k in range(p):
    Atemp = A_dist.local_array.copy() if j==k else np.empty_like(A_dist.local_array)
    Btemp = B_dist.local_array.copy() if i==k else np.empty_like(B_dist.local_array)
    if rank==0: print(k,"cast")
    row_comm.Bcast([Atemp, MPI.FLOAT], root=k)
    col_comm.Bcast([Btemp, MPI.FLOAT], root=k)
    c += Atemp @ Btemp
    if rank == 0: print(k,"after")

C_dist.local_array[:] = c
if rank==0: print(C_dist.asarray())