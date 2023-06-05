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

# Defining the global shape of the distributed array
global_shape = (10, 10)

###############################################################################
# Let's start by defining the
# class with the input parameters ``global_shape``,
# ``partition``, and ``axis``. Here's an example implementation of the class with ``axis=0``.
arr = pylops_mpi.DistributedArray(global_shape=global_shape,
                                  partition=pylops_mpi.Partition.SCATTER,
                                  axis=0)
arr[:] = np.arange(arr.local_shape[0] * arr.local_shape[1] * arr.rank,
                   arr.local_shape[0] * arr.local_shape[1] * (arr.rank + 1)).reshape(arr.local_shape)
pylops_mpi.plot_distributed_array(arr)

###############################################################################
# Below is an implementation to show how the global array is distributed along
# the second axis.
arr = pylops_mpi.DistributedArray(global_shape=global_shape,
                                  partition=pylops_mpi.Partition.SCATTER,
                                  axis=1)
arr[:] = np.arange(arr.local_shape[0] * arr.local_shape[1] * arr.rank,
                   arr.local_shape[0] * arr.local_shape[1] * (arr.rank + 1)).reshape(arr.local_shape)
pylops_mpi.plot_distributed_array(arr)

###############################################################################
# To convert a random NumPy array into a ``pylops_mpi.DistributedArray``,
# you can use the ``to_dist`` classmethod. This method allows you to distribute
# the array across multiple processes for parallel computation.
# Below is an example implementation depicting the same.
arr1 = pylops_mpi.DistributedArray.to_dist(np.random.normal(100, 100, global_shape))
arr2 = pylops_mpi.DistributedArray.to_dist(np.random.normal(300, 300, global_shape))
# plot local arrays
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
