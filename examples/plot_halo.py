#mpirun --oversubscribe -np 8 python3 plot_halo.py
import math
import numpy as np
import pylops_mpi
from pylops_mpi.basicoperators.Halo import MPIHalo, ScatterType, local_block_split, BoundaryType
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

gdim = (8, 8, 8)
p_prime = int(math.pow(size, 1/3))
g_shape = (p_prime,p_prime,p_prime)

halo_op = MPIHalo(
    dims=gdim,
    halo=1,
    scatter=ScatterType.BLOCK,
    proc_grid_shape=g_shape,
    boundary_mode=BoundaryType.ZERO,
    comm=comm
)

x_data  = np.arange(np.prod(gdim)).astype(np.float64).reshape(gdim)
x_slice = local_block_split(gdim, comm, g_shape)
x_local = x_data[x_slice]
x_dist  = pylops_mpi.DistributedArray(global_shape=np.prod(gdim),
                                     local_shapes=comm.allgather(np.prod(x_local.shape)),
                                     base_comm=comm,
                                     partition=pylops_mpi.Partition.SCATTER)

x_dist.local_array[:] = x_local.flatten()

x_with_halo = halo_op @ x_dist
print(rank, x_with_halo.local_array.reshape(gdim[0]//p_prime + 2, gdim[1]//p_prime + 2, gdim[2]//p_prime + 2))
x_extracted = halo_op.H @ x_with_halo