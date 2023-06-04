from matplotlib import pyplot as plt
import numpy as np
import pylops_mpi

plt.close("all")
np.random.seed(42)

global_shape = (10, 10)

# Distribution along axis = 0
distributed_array = pylops_mpi.DistributedArray(global_shape=global_shape,
                                                partition=pylops_mpi.Partition.SCATTER,
                                                axis=0)
start = distributed_array.local_shape[0] * distributed_array.local_shape[1] * distributed_array.rank
end = distributed_array.local_shape[0] * distributed_array.local_shape[1] * (distributed_array.rank + 1)
distributed_array[:] = np.arange(start, end).reshape(distributed_array.local_shape)
pylops_mpi.plot_distributed_array(distributed_array)

# Distribution along axis = 1
distributed_array = pylops_mpi.DistributedArray(global_shape=global_shape,
                                                partition=pylops_mpi.Partition.SCATTER,
                                                axis=1)

start = distributed_array.local_shape[0] * distributed_array.local_shape[1] * distributed_array.rank
end = distributed_array.local_shape[0] * distributed_array.local_shape[1] * (distributed_array.rank + 1)
distributed_array[:] = np.arange(start, end).reshape(distributed_array.local_shape)
pylops_mpi.plot_distributed_array(distributed_array)

# Example of ``to_dist``
arr1 = pylops_mpi.DistributedArray.to_dist(np.random.normal(100, 100, (10, 10)))
arr2 = pylops_mpi.DistributedArray.to_dist(np.random.normal(300, 300, (10, 10)))
pylops_mpi.plot_distributed_array(arr1)
pylops_mpi.plot_distributed_array(arr2)

# Element-wise Addition
sum_arr = arr1 + arr2

# Element-wise Subtraction
diff_arr = arr1 - arr2

# Element-wise Multiplication
mult_array = arr1 * arr2
