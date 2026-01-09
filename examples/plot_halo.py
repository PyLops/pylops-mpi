#mpirun --oversubscribe -np 8 python3 plot_halo.py
import math
import numpy as np
import pylops
import pylops_mpi
from pylops_mpi.basicoperators.Halo import MPIHalo, ScatterType, local_block_split, BoundaryType
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

gdim = (8, 8, 8)
p_prime = int(math.pow(size, 1/3))
g_shape = (p_prime, p_prime, p_prime) # number of partitions over each axis
# need to make sure the overal number of partitions matches the number of ranks
assert np.prod(g_shape) == size
print('g_shape', g_shape)

# so far seem to require gdim to be divisible by g_shape (in other words cannot hav
# blocks of different size)

halo = 1
halo_op = MPIHalo(
    dims=gdim,
    halo=halo,
    scatter=ScatterType.BLOCK,
    proc_grid_shape=g_shape,
    boundary_mode=BoundaryType.ZERO,
    comm=comm
)

x_data  = np.arange(np.prod(gdim)).astype(np.float64).reshape(gdim)
x_slice = local_block_split(gdim, comm, g_shape)
x_local = x_data[x_slice]
x_local_shape = x_local.shape
x_dist  = pylops_mpi.DistributedArray(global_shape=np.prod(gdim),
                                      local_shapes=comm.allgather(np.prod(x_local.shape)),
                                      base_comm=comm,
                                      partition=pylops_mpi.Partition.SCATTER)

x_dist.local_array[:] = x_local.flatten()

x_with_halo = halo_op @ x_dist
print(rank, x_local.shape, x_dist.local_array.reshape(x_dist.local_shape).shape, 
      x_with_halo.local_array.reshape(gdim[0]//p_prime + 2, gdim[1]//p_prime + 2, gdim[2]//p_prime + 2).shape)
x_extracted = halo_op.H @ x_with_halo

# Diagonal operator acting also on the halo, then halo is remobed
local_shape_with_halo = np.prod([s + 2 * halo for s in x_local.shape]) # *2 as halo is on both sides,
DOp = pylops.Diagonal(2 * np.ones(local_shape_with_halo))
DOp_dist = pylops_mpi.MPIBlockDiag([DOp, ])

Op_dist = halo_op.H @ DOp_dist @ halo_op
y_dist = Op_dist @ x_dist

assert np.allclose(2 * x_dist.local_array, y_dist.local_array)

# Derivative operator
for axis in [0, 1, 2]:
    local_shape_with_halo = [s + 2 * halo for s in x_local.shape]
    print('local_shape_with_halo', local_shape_with_halo)
    DOp = pylops.FirstDerivative(dims=local_shape_with_halo, axis=axis)
    DOp_dist = pylops_mpi.MPIBlockDiag([DOp, ])

    Op_dist = halo_op.H @ DOp_dist @ halo_op
    y_dist = Op_dist @ x_dist

    DOp1 = pylops.FirstDerivative(dims=gdim, axis=axis)
    y = DOp1 @ x_data

    if rank == 0:
        print('y', y)
        print('x_dist', x_dist.local_array.reshape(x_local_shape))
        print('y_dist', y_dist.local_array.reshape(x_local_shape)[1:-1,1:-1,1:-1])
        print('y', y[x_slice][1:-1,1:-1,1:-1])

    # Check that derivative on entire 3d object is the same as the one on blocks with halo (remove
    # borders as on the edge the derivatives of the derivative operator are 0 but those from the haloed
    # object are computed using the halo valu - here 0)
    assert np.allclose(y[x_slice][1:-1,1:-1,1:-1],
                    y_dist.local_array.reshape(x_local_shape)[1:-1,1:-1,1:-1])