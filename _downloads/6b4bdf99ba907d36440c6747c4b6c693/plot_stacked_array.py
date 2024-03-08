"""
Stacked Array
=========================
This example shows how to use the :py:class:`pylops_mpi.StackedDistributedArray`.
This class provides a way to combine and act on multiple :py:class:`pylops_mpi.DistributedArray`
within the same program. This is very useful in scenarios where an array can be logically
divided in subarrays and each of them lends naturally to distribution across multiple processes in
a parallel computing environment.
"""

from matplotlib import pyplot as plt
import numpy as np
from mpi4py import MPI

import pylops
import pylops_mpi

plt.close("all")
np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

###############################################################################
# Let's start by defining two distributed array
subarr1 = pylops_mpi.DistributedArray(global_shape=size * 10,
                                      partition=pylops_mpi.Partition.SCATTER,
                                      axis=0)
subarr2 = pylops_mpi.DistributedArray(global_shape=size * 4,
                                      partition=pylops_mpi.Partition.SCATTER,
                                      axis=0)
# Filling the local arrays
subarr1[:], subarr2[:] = 1, 2

###############################################################################
# We combine them into a single
# :py:class:`pylops_mpi.StackedDistributedArray` object.
arr1 = pylops_mpi.StackedDistributedArray([subarr1, subarr2])
if rank == 0:
    print('Stacked array:', arr1)

# Extract and print full array
full_arr1 = arr1.asarray()
if rank == 0:
    print('Full array:', full_arr1)

# Modify the part of the first array in rank0
if rank == 0:
    arr1[0][:] = 10
full_arr1 = arr1.asarray()
if rank == 0:
    print('Modified full array:', full_arr1)

###############################################################################
# Let's now create a second :py:class:`pylops_mpi.StackedDistributedArray` object
# and perform different mathematical operations on those two objects.
subarr1_ = pylops_mpi.DistributedArray(global_shape=size * 10,
                                       partition=pylops_mpi.Partition.SCATTER,
                                       axis=0)
subarr2_ = pylops_mpi.DistributedArray(global_shape=size * 4,
                                       partition=pylops_mpi.Partition.SCATTER,
                                       axis=0)
# Filling the local arrays
subarr1_[:], subarr2_[:] = 5, 6
arr2 = pylops_mpi.StackedDistributedArray([subarr1_, subarr2_])
if rank == 0:
    print('Stacked array 2:', arr2)

full_arr2 = arr2.asarray()
if rank == 0:
    print('Full array2:', full_arr2)

###############################################################################
# **Negation**
neg_arr = -arr1
full_neg_arr = neg_arr.asarray()
if rank == 0:
    print('Negated full array:', full_neg_arr)

###############################################################################
# **Element-wise Addition**
sum_arr = arr1 + arr2
full_sum_arr = sum_arr.asarray()
if rank == 0:
    print('Summed full array:', full_sum_arr)

###############################################################################
# **Element-wise Subtraction**
sub_arr = arr1 - arr2
full_sub_arr = sub_arr.asarray()
if rank == 0:
    print('Subtracted full array:', full_sub_arr)

###############################################################################
# **Multiplication**
mult_arr = arr1 * arr2
full_mult_arr = mult_arr.asarray()
if rank == 0:
    print('Multipled full array:', full_mult_arr)

###############################################################################
# **Dot-product**
dot_arr = arr1.dot(arr2)
if rank == 0:
    print('Dot-product:', dot_arr)
    print('Dot-product (np):', np.dot(full_arr1, full_arr2))

###############################################################################
# **Norms**
l0norm = arr1.norm(0)
l1norm = arr1.norm(1)
l2norm = arr1.norm(2)
linfnorm = arr1.norm(np.inf)

if rank == 0:
    print('L0 norm', l0norm, np.linalg.norm(full_arr1, 0))
    print('L1 norm', l1norm, np.linalg.norm(full_arr1, 1))
    print('L2 norm', l2norm, np.linalg.norm(full_arr1, 2))
    print('Linf norm', linfnorm, np.linalg.norm(full_arr1, np.inf))

###############################################################################
# Now that we have a way to stack multiple :py:class:`pylops_mpi.StackedDistributedArray` objects,
# let's see how we can apply operators to them. More specifically this can be
# done using the :py:class:`pylops_mpi.MPIStackedVStack` operator that takes multiple
# :py:class:`pylops_mpi.MPILinearOperator` objects, each acting on one specific
# distributed array
x = pylops_mpi.DistributedArray(global_shape=size * 10,
                                partition=pylops_mpi.Partition.SCATTER,
                                axis=0)
# Filling the local arrays
x[:] = 1.

# Make stacked operator
mop1 = pylops_mpi.MPIBlockDiag([pylops.MatrixMult(np.ones((5, 10))), ])
mop2 = pylops_mpi.MPIBlockDiag([pylops.MatrixMult(2 * np.ones((8, 10))), ])
mop = pylops_mpi.MPIStackedVStack([mop1, mop2])

y = mop.matvec(x)
y_arr = y.asarray()
xadj = mop.rmatvec(y)
xadj_arr = xadj.asarray()

if rank == 0:
    print('StackedVStack y', y, y_arr, y_arr.shape)
    print('StackedVStack xadj', xadj, xadj_arr, xadj_arr.shape)

###############################################################################
# Finally, let's solve now an inverse problem using stacked arrays instead
# of distributed arrays
x0 = x.copy()
x0[:] = 0.
xinv = pylops_mpi.cgls(mop, y, x0=x0, niter=15, tol=1e-10, show=False)[0]
xinv_array = xinv.asarray()

if rank == 0:
    print('xinv_array', xinv_array)
