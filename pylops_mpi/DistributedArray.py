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
    dtype : :obj:`str`, optional
        Type of elements in input array. Defaults to ``float``.
    base_comm : :obj:`MPI.Comm`, optional
        MPI Communicator over which array is distributed. Defaults to ``MPI.COMM_WORLD``.
    type_part : :obj:`str`, optional
        Broadcast or Scatter the array. Defaults to ``S``.
    """

    def __new__(cls, global_shape: Union[Tuple, Integral], dtype: Optional[DTypeLike] = "float",
                base_comm: Optional[MPI.Comm] = MPI.COMM_WORLD, type_part: str = "S"):
        if type_part not in ["B", "S"]:
            raise ValueError("Should be either B or S")
        if isinstance(global_shape, Integral):
            global_shape = (global_shape,)
        # Broadcast the array
        if type_part == "B":
            local_shape = global_shape
        # Scatter the array
        else:
            local_shape = [((global_shape[0] // base_comm.Get_size()) + 1)
                           if base_comm.Get_rank() < (
                    global_shape[0] % base_comm.Get_size())
                           else global_shape[0] // base_comm.Get_size()] + list(global_shape[1:])
        # create an empty numpy local array
        arr = super().__new__(cls, local_shape, dtype)
        arr._global_shape = global_shape
        arr._local_shape = tuple(local_shape)
        arr._base_comm = base_comm
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
    def to_dist(cls, x: NDArray, base_comm: MPI.Comm = MPI.COMM_WORLD):
        """Convert A Global Array to a Distributed Array

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Global array.
        base_comm : :obj:`MPI.Comm`, optional
            Type of elements in input array. Defaults to ``MPI.COMM_WORLD``

        Returns
        ----------
        dist_array : :obj:`DistributedArray`
            Distributed Array of the Global Array

        """
        dist_array = DistributedArray(x.shape, x.dtype)
        local_shapes = np.append([0], base_comm.allgather(dist_array.local_shape))
        sum_shapes = np.cumsum(local_shapes)
        dist_array[:] = x[slice(sum_shapes[base_comm.Get_rank()], sum_shapes[base_comm.Get_rank() + 1], None)]
        return dist_array

    def __add__(self, x):
        return self.add(x)

    def __sub__(self, x):
        return self.add(-x)

    def __mul__(self, x):
        return self.multiply(x)

    def add(self, dist_array):
        """Distributed Addition of arrays
        """
        if self.shape != dist_array.shape:
            raise ValueError("Shape Mismatch")
        SumArray = DistributedArray(self.global_shape, dtype=self.dtype)
        SumArray[:] = self.local_array + dist_array.local_array
        return SumArray

    def multiply(self, dist_array):
        """Distributed Element-wise multiplication
        """
        if self.shape != dist_array.shape:
            raise ValueError("Shape Mismatch")
        ProductArray = DistributedArray(self.global_shape, self.dtype)
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
               f"processes={[i for i in range(self.base_comm.Get_size())]})> "

    def __str__(self):
        return f"<DistributedArray with global shape={self.global_shape}), local shape={self.local_shape}" \
               f", dtype={self.dtype}, " \
               f"processes={[i for i in range(self.base_comm.Get_size())]})> "

    # def broadcast(self, x: NDArray) -> None:
    #     """Broadcast the NumPy Array
    #
    #     Parameters
    #     ----------
    #     x : :obj:`np.ndarray`
    #         Array to be broadcasted
    #     """
    #     local_array = None
    #     if x is not None:
    #         local_array = np.empty(x.shape, dtype=self.dtype)
    #         if self.base_comm.Get_rank() == 0:
    #             local_array = x.astype(self.dtype)
    #     self.base_comm.bcast(local_array, root=0)
    #
    # def scatter(self, x: NDArray):
    #     """Scatter the NumPy Array
    #
    #     Parameters
    #     ----------
    #     x : :obj:`np.ndarray`
    #         Array to be scattered
    #     """
    #     send_data = None
    #     if self.base_comm.Get_rank() == 0:
    #         arrs = np.array_split(x, self.size)
    #         raveled = [np.ravel(arr) for arr in arrs]
    #         send_data = np.concatenate(raveled, dtype=self.dtype)
    #     recv_shape = list(x.shape)
    #     recv_shape_new = [((recv_shape[0] // self.size) + 1) if self.base_comm.Get_rank() < (
    #             recv_shape[0] % self.size)
    #                       else recv_shape[0] // self.size] + recv_shape[1:]
    #     self.local_array = np.empty(recv_shape_new, dtype=self.dtype)
    #     self.base_comm.Scatterv(send_data, self.local_array, root=0)
