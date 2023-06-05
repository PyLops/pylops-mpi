"""
    Plotting functions for DistributedArray
"""

from matplotlib import pyplot as plt
import numpy as np

from pylops_mpi import DistributedArray, Partition


# Plot how the global array is distributed among ranks
def plot_distributed_array(arr: DistributedArray) -> None:
    """Visualize distribution of the global array among different ranks.

    Parameters
    ----------
    arr : :obj:`pylops_mpi.DistributedArray`
        DistributedArray
    """
    if not isinstance(arr, DistributedArray):
        raise TypeError("Not a DistributedArray")
    if arr.partition is Partition.BROADCAST:
        raise NotImplementedError("Use Scatter for plot")
    dist_array = DistributedArray(global_shape=arr.global_shape,
                                  base_comm=arr.base_comm,
                                  partition=arr.partition, axis=arr.axis,
                                  dtype=arr.dtype)
    dist_array[:] = np.full(shape=dist_array.local_shape, fill_value=arr.rank)
    full_dist_arr = dist_array.asarray()
    full_arr = arr.asarray()
    if arr.rank == 0:
        figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                          figsize=(18, 5))
        ax1.matshow(full_arr, cmap='rainbow')
        ax1.set_title("Original Array")
        im2 = ax2.matshow(full_dist_arr, cmap='rainbow')
        ax2.set_title(f"Distributed over axis {arr.axis}")
        cbar = figure.colorbar(im2)
        cbar.set_ticks(np.arange(arr.size))
        cbar.set_label("Ranks")
        plt.tight_layout()


# Plot the local arrays of each rank
def plot_local_arrays(arr: DistributedArray, title: str = None) -> None:
    """Visualize the local arrays of the given DistributedArray

    Parameters
    ----------
    arr : :obj:`pylops_mpi.DistributedArray`
        DistributedArray
    title : :obj:`str`
        Main Title of the figure
    """
    global_gather = arr.base_comm.gather(arr.local_array, root=0)
    if arr.rank == 0:
        plt.figure(figsize=(18, 5))
        for i in range(arr.size):
            plt.subplot(2, arr.size, i + 1)
            plt.imshow(global_gather[i], cmap='rainbow')
            plt.xticks(np.arange(global_gather[i].shape[1]))
            plt.yticks(np.arange(global_gather[i].shape[0]))
            plt.title(f"Rank-{i}")
        plt.suptitle(title)
        plt.tight_layout()

