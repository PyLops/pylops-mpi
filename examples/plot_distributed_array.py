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
global_shape = (10, 5)

###############################################################################
# Let's start by defining the
# class with the input parameters ``global_shape``,
# ``partition``, and ``axis``. Here's an example implementation of the class with ``axis=0``.
arr = pylops_mpi.DistributedArray(global_shape=global_shape,
                                  partition=pylops_mpi.Partition.SCATTER,
                                  axis=0)
# Filling the local arrays
arr[:] = np.arange(arr.local_shape[0] * arr.local_shape[1] * arr.rank,
                   arr.local_shape[0] * arr.local_shape[1] * (arr.rank + 1)).reshape(arr.local_shape)
pylops_mpi.plot_distributed_array(arr)

###############################################################################
# Below is an implementation to show how the global array is distributed along
# the second axis.
arr = pylops_mpi.DistributedArray(global_shape=global_shape,
                                  partition=pylops_mpi.Partition.SCATTER,
                                  axis=1)
# Filling the local arrays
arr[:] = np.arange(arr.local_shape[0] * arr.local_shape[1] * arr.rank,
                   arr.local_shape[0] * arr.local_shape[1] * (arr.rank + 1)).reshape(arr.local_shape)
pylops_mpi.plot_distributed_array(arr)

###############################################################################
# To convert a random NumPy array into a ``pylops_mpi.DistributedArray``,
# you can use the ``to_dist`` classmethod. This method allows you to distribute
# the array across multiple processes for parallel computation.
# Below is an example implementation depicting the same.
n = global_shape[0] * global_shape[1]
# Array to be distributed
array = np.arange(n) / float(n)
arr1 = pylops_mpi.DistributedArray.to_dist(x=array.reshape(global_shape), axis=1)
array = array / 2.0
arr2 = pylops_mpi.DistributedArray.to_dist(x=array.reshape(global_shape), axis=1)
# plot local arrays
pylops_mpi.plot_local_arrays(arr1, "Distributed Array - 1", vmin=0, vmax=1)
pylops_mpi.plot_local_arrays(arr2, "Distributed Array - 2", vmin=0, vmax=1)

###############################################################################
# **Element-wise Addition** - Each process operates on its local portion of
# the array and adds the corresponding elements together.
sum_arr = arr1 + arr2
pylops_mpi.plot_local_arrays(sum_arr, "Addition", vmin=0, vmax=1)

###############################################################################
# **Element-wise Subtraction** - Each process operates on its local portion
# of the array and subtracts the corresponding elements together.
diff_arr = arr1 - arr2
pylops_mpi.plot_local_arrays(diff_arr, "Subtraction", vmin=0, vmax=1)

###############################################################################
# **Element-wise Multiplication** - Each process operates on its local portion
# of the array and multiplies the corresponding elements together.
mult_arr = arr1 * arr2
pylops_mpi.plot_local_arrays(mult_arr, "Multiplication", vmin=0, vmax=1)
