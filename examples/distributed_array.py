import numpy as np

from pylops_mpi.DistributedArray import DistributedArray

# For global arrays use to_dist
distr = DistributedArray.to_dist(np.random.randint(1, 10, (100, )))
print(distr.local_shape, distr.local_array)

# Broadcast
arr = DistributedArray(global_shape=(1000, 100), dtype=int, partition="B")
arr[:] = np.random.normal(10, 1, arr.local_shape)

# Scatter
arr1 = DistributedArray(global_shape=(1000, 100), dtype=float, partition="S")
arr1[:] = np.random.normal(108, 2, arr1.local_shape)

# Distributed Array of ones and zeroes
arr2 = DistributedArray(global_shape=(1000, 100), dtype=int)
arr3 = DistributedArray(global_shape=(1000, 100), dtype=int)
arr2[:] = 0
arr3[:] = 1
