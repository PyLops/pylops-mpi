"""
Distributed Array
=========================
This example shows how to use the :py:class:`pylops_mpi.DistributedArray`.
This class provides a way to distribute arrays across multiple processes in
a parallel computing environment.
"""

from matplotlib import pyplot as plt
import numpy as np
import pylops_mpi

plt.close("all")
np.random.seed(42)

global_shape = (10, 10)


def fill_arrays(arr: pylops_mpi.DistributedArray):
    start = arr.local_shape[0] * arr.local_shape[1] * arr.rank
    end = arr.local_shape[0] * arr.local_shape[1] * (arr.rank + 1)
    arr[:] = np.arange(start, end).reshape(arr.local_shape)
    return arr


###############################################################################
# We use the DistributedArray class with input parameters `global_shape`,
# `partition` and `axis`.
distributed_array = pylops_mpi.DistributedArray(global_shape=global_shape,
                                                partition=pylops_mpi.Partition.SCATTER,
                                                axis=0)
pylops_mpi.plot_distributed_array(fill_arrays(distributed_array))


###############################################################################
# Here we use `axis` = 1
distributed_array = pylops_mpi.DistributedArray(global_shape=global_shape,
                                                partition=pylops_mpi.Partition.SCATTER,
                                                axis=1)
pylops_mpi.plot_distributed_array(fill_arrays(distributed_array))


###############################################################################
# Convert a random numpy array to a `pylops_mpi.DistributedArray`.
arr1 = pylops_mpi.DistributedArray.to_dist(np.random.normal(100, 100, global_shape))
arr2 = pylops_mpi.DistributedArray.to_dist(np.random.normal(300, 300, global_shape))
pylops_mpi.plot_local_arrays(arr1, "Distributed Array - 1")
pylops_mpi.plot_local_arrays(arr2, "Distributed Array - 2")

###############################################################################
# Element-wise Addition
sum_arr = arr1 + arr2
pylops_mpi.plot_local_arrays(sum_arr, "Addition")


###############################################################################
# Element-wise Subtraction
diff_arr = arr1 - arr2
pylops_mpi.plot_local_arrays(diff_arr, "Subtraction")


###############################################################################
# Element-wise Multiplication
mult_array = arr1 * arr2
pylops_mpi.plot_local_arrays(mult_array, "Multiplication")
