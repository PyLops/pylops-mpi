import numpy as np
from typing import Optional, Union, Tuple
from numbers import Integral
from mpi4py import MPI
from enum import Enum

from pylops.utils import DTypeLike, NDArray


class Partition(Enum):
    r"""Enum class

    Distributing data among different processes.

    - ``BROADCAST``: Distributes data to all processes.
    - ``SCATTER``: Distributes unique portions to each process.
    """
    BROADCAST = "Broadcast"
    SCATTER = "Scatter"


def local_split(global_shape: Tuple, base_comm: MPI.Comm,
                partition: Partition, axis: int):
    """To get the local shape from the global shape

    Parameters
    ----------
    global_shape : :obj:`tuple`
        Shape of the global array.
    base_comm : :obj:`MPI.Comm`
        Base MPI Communicator.
    partition : :obj:`Partition`
        Type of partition.
    axis : :obj:`int`
        Axis of distribution

    Returns
    -------
    local_shape : :obj:`tuple`
        Shape of the local array.
    """
    if partition == Partition.BROADCAST:
        local_shape = global_shape
    # Split the array
    else:
        local_shape = list(global_shape)
        if base_comm.Get_rank() < (global_shape[axis] % base_comm.Get_size()):
            local_shape[axis] = global_shape[axis] // base_comm.Get_size() + 1
        else:
            local_shape[axis] = global_shape[axis] // base_comm.Get_size()
    return tuple(local_shape)


class DistributedArray:
    r"""Distributed Numpy Arrays

    Multidimensional NumPy-like distributed arrays.
    It brings NumPy arrays to high-performance computing.

    Parameters
    ----------
    global_shape : :obj:`tuple` or :obj:`int`
        Shape of the global array.
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        MPI Communicator over which array is distributed.
        Defaults to ``mpi4py.MPI.COMM_WORLD``.
    partition : :obj:`Partition`, optional
        Broadcast or Scatter the array. Defaults to ``Partition.SCATTER``.
    dtype : :obj:`str`, optional
        Type of elements in input array. Defaults to ``numpy.float64``.
    axis : :obj:`int`, optional
        Axis along which distribution occurs. Defaults to ``0``.
    """

    def __init__(self, global_shape: Union[Tuple, Integral],
                 base_comm: Optional[MPI.Comm] = MPI.COMM_WORLD,
                 partition: Partition = Partition.SCATTER, axis: int = 0,
                 dtype: Optional[DTypeLike] = np.float64):
        if isinstance(global_shape, Integral):
            global_shape = (global_shape,)
        if len(global_shape) <= axis:
            raise IndexError(f"Axis {axis} out of range for DistributedArray "
                             f"of shape {global_shape}")
        if partition not in Partition:
            raise ValueError(f"Should be either {Partition.BROADCAST} "
                             f"or {Partition.SCATTER}")
        self.dtype = dtype
        self._global_shape = global_shape
        self._local_shape = local_split(global_shape, base_comm,
                                        partition, axis)
        self._base_comm = base_comm
        self._partition = partition
        self._axis = axis
        self._local_array = np.empty(shape=self.local_shape, dtype=self.dtype)

    def __getitem__(self, index):
        return self.local_array[index]

    def __setitem__(self, index, value):
        self.local_array[index] = value

    @property
    def global_shape(self):
        """Global Shape of the array

        Returns
        -------
        global_shape : :obj:`tuple`
        """
        return self._global_shape

    @property
    def base_comm(self):
        """Base MPI Communicator

         Returns
        -------
        base_comm : :obj:`MPI.Comm`
        """
        return self._base_comm

    @property
    def local_shape(self):
        """Local Shape of the Distributed array

        Returns
        -------
        local_shape : :obj:`tuple`
        """
        return self._local_shape

    @property
    def local_array(self):
        """View of the Local Array

        Returns
        -------
        local_array : :obj:`numpy.ndarray`
        """
        return self._local_array

    @property
    def rank(self):
        """Rank of the current process

        Returns
        -------
        rank : :obj:`int`
        """
        return self.base_comm.Get_rank()

    @property
    def size(self):
        """Total number of processes
        Size of parallel environment

        Returns
        -------
        size : :obj:`int`
        """
        return self.base_comm.Get_size()

    @property
    def axis(self):
        """Axis along which distribution occurs

        Returns
        -------
        axis : :obj:`int`
        """
        return self._axis

    @property
    def ndim(self):
        """Number of dimensions of the global array

        Returns
        -------
        ndim : :obj:`int`
        """
        return len(self.global_shape)

    @property
    def partition(self):
        """Type of Distribution

        Returns
        -------
        partition_type : :obj:`str`
        """
        return self._partition

    def asarray(self):
        """Global view of the array

        Gather all the local arrays

        Returns
        -------
        final_array : :obj:`numpy.ndarray`
            Global Array gathered at all ranks
        """
        # Since the global array was replicated at all ranks
        if self.partition == Partition.BROADCAST:
            # Get only self.local_array.
            return self.local_array
        # Gather all the local arrays and apply concatenation.
        final_array = self.base_comm.allgather(self.local_array)
        return np.concatenate(final_array, axis=self.axis)

    @classmethod
    def to_dist(cls, x: NDArray,
                base_comm: MPI.Comm = MPI.COMM_WORLD,
                partition: Partition = Partition.SCATTER,
                axis: int = 0):
        """Convert A Global Array to a Distributed Array

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Global array.
        base_comm : :obj:`MPI.Comm`, optional
            Type of elements in input array. Defaults to ``MPI.COMM_WORLD``
        partition : :obj:`Partition`, optional
            Distributes the array, Defaults to ``Partition.Scatter``.
        axis : :obj:`int`, optional
            Axis of Distribution
        Returns
        ----------
        dist_array : :obj:`DistributedArray`
            Distributed Array of the Global Array
        """
        dist_array = DistributedArray(global_shape=x.shape,
                                      base_comm=base_comm,
                                      partition=partition,
                                      axis=axis,
                                      dtype=x.dtype)
        if partition == Partition.BROADCAST:
            dist_array[:] = x
        else:
            slices = [slice(None)] * x.ndim
            local_shapes = np.append([0], base_comm.allgather(
                dist_array.local_shape[axis]))
            sum_shapes = np.cumsum(local_shapes)
            slices[axis] = slice(sum_shapes[dist_array.rank],
                                 sum_shapes[dist_array.rank + 1], None)
            dist_array[:] = x[tuple(slices)]
        return dist_array

    def _check_partition_shape(self, dist_array):
        """Check Partition and Local Shape of the Array
        """
        if self.partition != dist_array.partition:
            raise ValueError("Partition of both the arrays must be same")
        if self.local_shape != dist_array.local_shape:
            raise ValueError(f"Local Array Shape Mismatch - "
                             f"{self.local_shape} != {dist_array.local_shape}")

    def _allreduce(self, send_buf, recv_buf=None, op: MPI.Op = MPI.SUM):
        """MPI Allreduce operation
        """
        if recv_buf is None:
            return self.base_comm.allreduce(send_buf, op)
        # For MIN and MAX which require recv_buf
        self.base_comm.Allreduce(send_buf, recv_buf, op)
        return recv_buf

    def __neg__(self):
        arr = DistributedArray(global_shape=self.global_shape,
                               base_comm=self.base_comm,
                               partition=self.partition,
                               axis=self.axis,
                               dtype=self.dtype)
        arr[:] = -self.local_array
        return arr

    def __add__(self, x):
        return self.add(x)

    def __sub__(self, x):
        return self.__add__(-x)

    def __mul__(self, x):
        return self.multiply(x)

    def add(self, dist_array):
        """Distributed Addition of arrays
        """
        self._check_partition_shape(dist_array)
        SumArray = DistributedArray(global_shape=self.global_shape,
                                    dtype=self.dtype, partition=self.partition,
                                    axis=self.axis)
        SumArray[:] = self.local_array + dist_array.local_array
        return SumArray

    def multiply(self, dist_array):
        """Distributed Element-wise multiplication
        """
        self._check_partition_shape(dist_array)
        ProductArray = DistributedArray(global_shape=self.global_shape,
                                        dtype=self.dtype,
                                        partition=self.partition,
                                        axis=self.axis)
        ProductArray[:] = self.local_array * dist_array.local_array
        return ProductArray

    def dot(self, dist_array):
        """Distributed Dot Product
        """
        self._check_partition_shape(dist_array)
        # Flatten the local arrays and calculate dot product
        return self._allreduce(np.dot(self.local_array.flatten(), dist_array.local_array.flatten()))

    def _compute_vector_norm(self, local_array: NDArray,
                             axis: int, ord: Optional[int] = None):
        """Compute Vector norm using MPI

        Parameters
        ----------
        local_array : :obj:`numpy.ndarray`
            Local Array at each rank
        axis : :obj:`int`
            Axis along which norm is computed
        ord : :obj:`int`, optional
            Order of the norm
        """
        # Compute along any axis
        ord = 2 if ord is None else ord
        if local_array.ndim == 1:
            recv_buf = np.empty(shape=1, dtype=np.float64)
        else:
            global_shape = list(self.global_shape)
            global_shape[axis] = 1
            recv_buf = np.empty(shape=global_shape, dtype=np.float64)
        if ord in ['fro', 'nuc']:
            raise ValueError(f"norm-{ord} not possible for vectors")
        elif ord == 0:
            # Count non-zero then sum reduction
            recv_buf = self._allreduce(np.count_nonzero(local_array, axis=axis).astype(np.float64))
        elif ord == np.inf:
            # Calculate max followed by max reduction
            recv_buf = self._allreduce(np.max(np.abs(local_array), axis=axis).astype(np.float64),
                                       recv_buf, op=MPI.MAX)
            recv_buf = np.squeeze(recv_buf, axis=axis)
        elif ord == -np.inf:
            # Calculate min followed by min reduction
            recv_buf = self._allreduce(np.min(np.abs(local_array), axis=axis).astype(np.float64),
                                       recv_buf, op=MPI.MIN)
            recv_buf = np.squeeze(recv_buf, axis=axis)

        else:
            recv_buf = self._allreduce(np.sum(np.abs(np.float_power(local_array, ord)), axis=axis))
            recv_buf = np.power(recv_buf, 1. / ord)
        return recv_buf

    def norm(self, ord: Optional[int] = None,
             axis: int = -1):
        """Distributed numpy.linalg.norm method

        Parameters
        ----------
        ord : :obj:`int`, optional
            Order of the norm.
        axis : :obj:`int`, optional
            Axis along which vector norm needs to be computed. Defaults to ``-1``
        """
        if axis == -1:
            # Flatten the local arrays and calculate norm
            return self._compute_vector_norm(self.local_array.flatten(), axis=0, ord=ord)
        if axis != self.axis:
            raise NotImplementedError("Choose axis along the partition.")
        # Calculate vector norm along the axis
        return self._compute_vector_norm(self.local_array, axis=self.axis, ord=ord)

    def conj(self):
        """Distributed conj() method
        """
        conj = DistributedArray(global_shape=self.global_shape, dtype=self.dtype)
        conj[:] = self.local_array.conj()
        return conj

    def copy(self):
        """Creates a copy of the DistributedArray
        """
        arr = DistributedArray(global_shape=self.global_shape, dtype=self.dtype)
        arr[:] = self.local_array
        return arr

    def __repr__(self):
        return f"<DistributedArray with global shape={self.global_shape}), " \
               f"local shape={self.local_shape}" \
               f", dtype={self.dtype}, " \
               f"processes={[i for i in range(self.size)]})> "
