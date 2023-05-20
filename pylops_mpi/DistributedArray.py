import numpy as np
from typing import Optional, Union

from mpi4py import MPI

from pylops.utils import DTypeLike, NDArray, ShapeLike


class DistributedArray:
    """Distributed Numpy Arrays
    Multidimensional NumPy-like distributed arrays.
    It brings NumPy arrays to high-performance computing

    Attributes
    ----------
    shape : :obj:`tuple(int, int)`
        Shape of the global array
    dtype : :obj:`str`, optional
        Type of elements in input array. Defaults to ``float``.
    """

    def __init__(self, shape: Optional[ShapeLike] = None,
                 dtype: Optional[DTypeLike] = 'float') -> None:
        self.dtype = dtype
        self.shape = shape
        self.base_comm = MPI.COMM_WORLD
        self.size = self.base_comm.Get_size()
        self.rank = self.base_comm.Get_rank()
        self.local_array = None

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, new_shape):
        self._shape = new_shape

    @shape.deleter
    def shape(self):
        del self._shape

    def assign(self, x: NDArray):
        pass

    def broadcast(self, x: NDArray) -> None:
        """Broadcast the NumPy Array

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Array to be broadcasted
        """
        if x is not None:
            self.local_array = np.empty(x.shape, dtype=self.dtype)
            if self.rank == 0:
                self.local_array = x.astype(self.dtype)
        self.base_comm.Bcast(self.local_array, root=0)

    def scatter(self, x: NDArray):
        """Scatter the NumPy Array

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Array to be scattered
        """
        send_data = None
        if self.rank == 0:
            arrs = np.array_split(x, self.size)
            raveled = [np.ravel(arr) for arr in arrs]
            send_data = np.concatenate(raveled, dtype=self.dtype)
        recv_shape = list(x.shape)
        recv_shape_new = [((recv_shape[0] // self.size) + 1) if self.rank < (
                recv_shape[0] % self.size)
                          else recv_shape[0] // self.size] + recv_shape[1:]
        self.local_array = np.empty(recv_shape_new, dtype=self.dtype)
        self.base_comm.Scatterv(send_data, self.local_array, root=0)

    def get_local_arrays(self):
        """Gather all the local arrays
        Returns
        -------
        final_array : :obj:`np.ndarray`
            Global Array gathered at all ranks
        """

        final_array = self.base_comm.allgather(self.local_array)
        return final_array

    def __add__(self, x):
        return self.add(x)

    def __mul__(self, x):
        return self.multiply(x)

    def add(self, dist_array):
        """Distributed Addition of arrays
        """
        if self.shape != dist_array.shape:
            raise ValueError("Shape Mismatch")
        recv_buff = np.empty(shape=self.shape, dtype=self.dtype)
        SumArray = DistributedArray(recv_buff.shape, recv_buff.dtype)
        SumArray.local_array = self.local_array + dist_array.local_array
        return SumArray

    def multiply(self, dist_array):
        """Distributed Element-wise multiplication
        """
        if self.shape != dist_array.shape:
            raise ValueError("Shape Mismatch")
        recv_buff = np.empty(shape=self.shape, dtype=self.dtype)
        ProductArray = DistributedArray(recv_buff.shape, recv_buff.dtype)
        ProductArray.local_array = self.local_array * dist_array.local_array
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
        return f"<DistributedArray with shape={self.shape})" \
               f", dtype={self.dtype}, " \
               f"processes={[i for i in range(self.size)]})> "
