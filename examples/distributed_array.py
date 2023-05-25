import numpy as np

from pylops_mpi.DistributedArray import DistributedArray

# Broadcast
arr = DistributedArray(global_shape=(1000, 100), dtype=int, type_part="B")
arr[:] = np.random.normal(10, 1, arr.shape)

# Scatter
arr1 = DistributedArray(global_shape=(1000, 100), dtype=float, type_part="S")
arr1[:] = np.random.normal(108, 2, arr1.shape)

# Distributed Array of ones and zeroes
arr2 = DistributedArray(global_shape=(1000, 100), dtype=int)
arr3 = DistributedArray(global_shape=(1000, 100), dtype=int)
arr2[:] = 0
arr3[:] = 1
