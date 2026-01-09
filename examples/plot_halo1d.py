"""
Halo
====
This example demonstrates how to use the :py:class:`pylops_mpi.basicoperators.MPIHalo`
operator.

This operator is specifically design to extend pylops-mpi's capabilities in terms of
chunking of N-dimensional distributed arrays when solving inverse problems with 
pylops-mpi's operators and solvers. As a matter of fact, whilst
:py:class:`pylops_mpi.DistributedArray` allows one to create N-dimensional array 
distributed over one ``axis``, only 1-dimensional distributed arrays can be used 
with pylops-mpi's operators and solvers. 

Moreover, several operator require accessing neighbouring values, which for values at 
the edges of a local array belong to neighouring ranks. The process of obtaining ghost 
cells (or halos) from neighouring ranks is implemented in the ``add_ghost_cells`` method
of the :py:class:`pylops_mpi.DistributedArray` class; however, once again as the chunking
is so far limited to only one axis, also haloing happens in a single dimension.

The :py:class:`pylops_mpi.basicoperators.MPIHalo` operator allows to internally convert a
1-dimensional distributed array into a N-dimensional distributed array (whilst never
physically changing the actual shape of local arrays) and extracting a number of user-defined
ghost cells over all axes. Moreover, even for the case when we are interested to chunk the 
distributed array over a single axis, having an operator that can be chained to any other 
pre-existent operator may open doors to an easier implementation of new operators compared
to having access to ``add_ghost_cells`` method to be used directly in the matvec/rmatvec
implementation of an operator.

"""
from matplotlib import pyplot as plt
import math
import sys
import time
import numpy as np
import pylops
from pylops_mpi.basicoperators.Halo_new import ScatterType, local_block_split, BoundaryType
from mpi4py import MPI

import pylops_mpi

plt.close("all")
np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def pause(comm, t=4):
    sys.stdout.flush()
    comm.barrier()
    time.sleep(t)

###############################################################################
# Let’s start by consider a 1-dimensional distributed array. We are interested 
# to compute a first derivative: however, because we are required to use a 
# stencil of size >=2, at the edges of the local arrays we are required to borrow
# cells from neighbouring ranks. As already mentioned, the 
# :py:class:`pylops_mpi.basicoperators.MPIFirstDerivative` operator does that 
# under the hood using the ``add_ghost_cells`` method of the 
# :py:class:`pylops_mpi.DistributedArray` class; however, we will see how we
# can now achieve the same without having to re-implement the derivative operator
# in pylops-mpi. Instead, we will simply combine the 
# :py:class:`pylops.basicoperators.FirstDerivative` operator with the
# :py:class:`pylops_mpi.basicoperators.MPIBlockDiag` and 
# :py:class:`pylops_mpi.basicoperators.MPIHalo` operators.
#
# To begin with, let's however simply create and apply a
# :py:class:`pylops_mpi.basicoperators.MPIHalo` operator with a halo of 1 
# (apart from the edges of the global array, where no halo is added)

# Halo operator
nlocal = 64
n = nlocal * size
proc_grid_shape = (size, )  # number of partitions over each axis

halo = 1
edge = False
halo_op = pylops_mpi.basicoperators.Halo_new.MPIHalo(
    dims=(n, ),
    halo=halo,
    edge=edge,
    scatter=ScatterType.BLOCK,
    proc_grid_shape=proc_grid_shape,
    boundary_mode=BoundaryType.ZERO,
    comm=comm
)

# Global array
x = np.arange(n).astype(np.float64)

# Distributed array
x_dist = pylops_mpi.DistributedArray(
    global_shape=n,
    # local_shapes=comm.allgather(np.prod(x_local.shape)),
    base_comm=comm,
    partition=pylops_mpi.Partition.SCATTER)
x_slice = local_block_split((n, ), comm, proc_grid_shape)
x_local = x[x_slice]
x_dist.local_array[:] = x_local

# Apply halo
x_dist_halo = halo_op @ x_dist

if rank == 0:
    print('Halo with halo=1')
pause(comm, t=1)
print(f"Rank {rank} - shape before/after haloing: {x_dist.local_array.size}"
      f"/{x_dist_halo.local_array.size}")


###############################################################################
# Let’s now instead compute the derivative and compare it with the result of 
# the :py:class:`pylops_mpi.basicoperators.FirstDerivative` operator.

# Derivative operator (using Halo)
if rank == 0 or rank == size - 1:
    local_shape_with_halo = (x_local.size + 1, )
else:
    local_shape_with_halo = (x_local.size + 2, )
DOp = pylops.FirstDerivative(dims=local_shape_with_halo, axis=0)
DOp_dist = pylops_mpi.MPIBlockDiag([DOp, ])
Op_dist = halo_op.H @ DOp_dist @ halo_op

y_dist = Op_dist @ x_dist

# Derivative operator (original)
DOp1_dist = pylops_mpi.MPIFirstDerivative(dims=n)

y1_dist = DOp1_dist @ x_dist

assert np.allclose(
    y_dist.local_array,
    y1_dist.local_array)

###############################################################################
# So good so far, we can reproduce a 2-nd order, centered first derivative 
# by combining basic pylops and pylops-mpi operators. Let's see if we can do the
# same with a 5-th order derivative that requires having an halo of 2.

pause(comm, t=1)

halo = 2
edge = False
halo_op = pylops_mpi.basicoperators.Halo_new.MPIHalo(
    dims=(n, ),
    halo=halo,
    edge=edge,
    scatter=ScatterType.BLOCK,
    proc_grid_shape=proc_grid_shape,
    boundary_mode=BoundaryType.ZERO,
    comm=comm
)

# Apply halo
x_dist_halo = halo_op @ x_dist

if rank == 2:
    print('Halo with halo=2')
pause(comm, t=1)
print(f"Rank {rank} - shape before/after haloing: {x_dist.local_array.size}"
      f"/{x_dist_halo.local_array.size}")

# Derivative operator (using Halo)
DOp = pylops.FirstDerivative(dims=x_dist_halo.local_array.size, axis=0, order=5)
DOp_dist = pylops_mpi.MPIBlockDiag([DOp, ])
Op_dist = halo_op.H @ DOp_dist @ halo_op

y_dist = Op_dist @ x_dist

# Derivative operator (original)
DOp1_dist = pylops_mpi.MPIFirstDerivative(dims=n, order=5)

y1_dist = DOp1_dist @ x_dist

assert np.allclose(
    y_dist.local_array,
    y1_dist.local_array)