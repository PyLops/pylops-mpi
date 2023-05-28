import numpy as np
from typing import Optional, Union, Tuple
from numbers import Integral
from mpi4py import MPI

from pylops.utils import DTypeLike, NDArray


class DistributedArray(np.ndarray):
    """Distributed Numpy Arrays
    Multidimensional NumPy-like distributed arrays.
    It brings NumPy arrays to high-performance computing

    It extends the `numpy.ndarray` class

    Attributes
    ----------
    global_shape : :obj:`tuple`
        Shape of the global array.
    base_comm : :obj:`MPI.Comm`, optional
        MPI Communicator over which array is distributed. Defaults to ``MPI.COMM_WORLD``.
    partition : :obj:`str`, optional
        Broadcast or Scatter the array. Defaults to ``S``.
    dtype : :obj:`str`, optional
        Type of elements in input array. Defaults to ``float``.
    """

    def __new__(cls, global_shape: Union[Tuple, Integral], base_comm: Optional[MPI.Comm] = MPI.COMM_WORLD,
                partition: str = "S", dtype: Optional[DTypeLike] = np.float64):
        if partition not in ["B", "S"]:
            raise ValueError("Should be either B or S")
        if isinstance(global_shape, Integral):
            global_shape = (global_shape,)
        # Broadcast the array
        if partition == "B":
            local_shape = global_shape
        # Scatter the array
        else:
            local_shape = [((global_shape[0] // base_comm.Get_size()) + 1)
                           if base_comm.Get_rank() < (
                    global_shape[0] % base_comm.Get_size())
                           else global_shape[0] // base_comm.Get_size()] + list(global_shape[1:])
        # create an empty numpy local array
        arr = np.ndarray.__new__(cls, shape=local_shape, dtype=dtype)
        arr._global_shape = global_shape
        arr._local_shape = tuple(local_shape)
        arr._base_comm = base_comm
        arr._partition_ = partition
        arr._rank = base_comm.Get_rank()
        arr._size = base_comm.Get_size()
        return arr

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
        local_array : :obj:`np.ndarray`
        """
        return self.view(np.ndarray)

    @property
    def rank(self):
        """Rank of the current process

        Returns
        -------
        rank : :obj:`int`
        """
        return self._rank

    @property
    def size(self):
        """Total number of processes
        Size of parallel environment

        Returns
        -------
        size : :obj:`int`
        """
        return self._size

    @property
    def partition_(self):
        """Type of Distribution

        Returns
        -------
        partition_type : :obj:`str`
        """
        return self._partition_

    def get_local_arrays(self):
        """Gather all the local arrays
        Returns
        -------
        final_array : :obj:`np.ndarray`
            Global Array gathered at all ranks
        """
        final_array = self.base_comm.allgather(self.local_array)
        return final_array

    @classmethod
    def to_dist(cls, x: NDArray, base_comm: MPI.Comm = MPI.COMM_WORLD, partition: str = "S"):
        """Convert A Global Array to a Distributed Array

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Global array.
        base_comm : :obj:`MPI.Comm`, optional
            Type of elements in input array. Defaults to ``MPI.COMM_WORLD``
        partition : :obj:`str`, optional
            Distributes the array, Defaults to ``S``.
        Returns
        ----------
        dist_array : :obj:`DistributedArray`
            Distributed Array of the Global Array

        """
        dist_array = DistributedArray(global_shape=x.shape, base_comm=base_comm,
                                      partition=partition, dtype=x.dtype)
        local_shapes = np.append([0], dist_array.base_comm.allgather(dist_array.local_shape[0]))
        sum_shapes = np.cumsum(local_shapes)
        dist_array[:] = x[slice(sum_shapes[dist_array.rank], sum_shapes[dist_array.rank + 1], None)]
        return dist_array

    def __neg__(self):
        arr = DistributedArray(global_shape=self.global_shape, base_comm=self.base_comm,
                               partition=self.partition_, dtype=self.dtype)
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
        if self.partition_ != dist_array.partition_:
            raise ValueError("Partition of both the Distributed Array must be same")
        if self.shape != dist_array.shape:
            raise ValueError("Shape Mismatch")
        SumArray = DistributedArray(global_shape=self.global_shape,
                                    dtype=self.dtype, partition=self.partition_)
        SumArray[:] = self.local_array + dist_array.local_array
        return SumArray

    def multiply(self, dist_array):
        """Distributed Element-wise multiplication
        """
        if self.partition_ != dist_array.partition_:
            raise ValueError("Partition of both the Distributed Array must be same")
        if self.shape != dist_array.shape:
            raise ValueError("Shape Mismatch")
        ProductArray = DistributedArray(global_shape=self.global_shape,
                                        dtype=self.dtype, partition=self.partition_)
        ProductArray[:] = self.local_array * dist_array.local_array
        return ProductArray

    def dot(self, dist_array):
        """Distributed Dot Product
        """
        pass

    def norm(self, x: NDArray, ord: Union[int, None] = None, axis: Optional[int] = None):
        """Distributed np.linalg.norm method
        """
        pass

    def __repr__(self):
        return f"<DistributedArray with global shape={self.global_shape}), local shape={self.local_shape}" \
               f", dtype={self.dtype}, " \
               f"processes={[i for i in range(self.size)]})> "

    def __str__(self):
        return f"<DistributedArray with global shape={self.global_shape}), local shape={self.local_shape}" \
               f", dtype={self.dtype}, " \
               f"processes={[i for i in range(self.size)]})> "

