from enum import Enum
from numbers import Integral
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from mpi4py import MPI
from pylops.utils import DTypeLike, NDArray, deps
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import get_array_module, get_module, get_module_name
from pylops_mpi.utils import deps as pylops_mpi_deps

cupy_message = deps.cupy_import("the DistributedArray module")
nccl_message = pylops_mpi_deps.nccl_import("the DistributedArray module")

if nccl_message is None and cupy_message is None:
    from pylops_mpi.utils._nccl import nccl_allgather, nccl_allreduce, nccl_asarray, nccl_bcast, nccl_split
    from cupy.cuda.nccl import NcclCommunicator
else:
    NcclCommunicator = Any


class Partition(Enum):
    r"""Enum class

    Distributing data among different processes.

    - ``BROADCAST``: Distributes data to all processes
      (ensuring that data is kept consistent across processes)
    - ``UNSAFE_BROADCAST``: Distributes data to all processes
      (without ensuring that data is kept consistent across processes)
    - ``SCATTER``: Distributes unique portions to each process.
    """
    BROADCAST = "Broadcast"
    UNSAFE_BROADCAST = "UnsafeBroadcast"
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
    if partition in [Partition.BROADCAST, Partition.UNSAFE_BROADCAST]:
        local_shape = global_shape
    # Split the array
    else:
        # TODO: Both NCCL and MPI have to use MPI.COMM_WORLD for shape splitting
        # So it was decided to make it explicit. This will make base_comm unused.
        local_shape = list(global_shape)
        if MPI.COMM_WORLD.Get_rank() < (global_shape[axis] % MPI.COMM_WORLD.Get_size()):
            local_shape[axis] = global_shape[axis] // MPI.COMM_WORLD.Get_size() + 1
        else:
            local_shape[axis] = global_shape[axis] // MPI.COMM_WORLD.Get_size()
    return tuple(local_shape)


def subcomm_split(mask, base_comm: MPI.Comm = MPI.COMM_WORLD):
    """Create new communicators based on mask

    This method creates new NCCL communicators based on ``mask``.
    Contrary to MPI, NCCL does not provide support for splitting of a communicator 
    in multiple subcommunicators; this is therefore handled explicitly by this method.

    Parameters
    ----------
    mask : :obj:`list`
        Mask defining subsets of ranks to consider when performing 'global'
        operations on the distributed array such as dot product or norm.

    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        A Communicator over which array is distributed
        Defaults to ``mpi4py.MPI.COMM_WORLD``.

    Returns:
    -------
    sub_comm : :obj:`mpi4py.MPI.Comm` or :obj:`cupy.cuda.nccl.NcclCommunicator`
        Subcommunicator according to mask
    """
    if isinstance(base_comm, MPI.Comm):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        sub_comm = comm.Split(color=mask[rank], key=rank)
    else:
        sub_comm = nccl_split(mask)
    return sub_comm


class DistributedArray:
    r"""Distributed Numpy Arrays

    Multidimensional NumPy-like distributed arrays.
    It brings NumPy arrays to high-performance computing.

    .. warning:: When setting the partition of the DistributedArray to
        :obj:`pylops_mpi.Partition.BROADCAST`, it is crucial to be aware
        that any attempts to make arrays different from rank to rank will be
        overwritten by the actions of rank 0. This is accomplished internally
        by broadcasting the content of rank 0 every time a modification of
        the array is attempted. Such a behaviour does however incur a cost
        as communication may be not needed if the user ensures not to modify
        the content of the array in different ranks in a different way. To
        avoid broadcasting, one can use :obj:`pylops_mpi.Partition.UNSAFE_BROADCAST`
        instead.

    Parameters
    ----------
    global_shape : :obj:`tuple` or :obj:`int`
        Shape of the global array.
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        MPI Communicator over which array is distributed.
        Defaults to ``mpi4py.MPI.COMM_WORLD``.
    partition : :obj:`Partition`, optional
        Broadcast, UnsafeBroadcast, or Scatter the array. Defaults to ``Partition.SCATTER``.
    axis : :obj:`int`, optional
        Axis along which distribution occurs. Defaults to ``0``.
    local_shapes : :obj:`list`, optional
        List of tuples or integers representing local shapes at each rank.
    mask : :obj:`list`, optional
        Mask defining subsets of ranks to consider when performing 'global'
        operations on the distributed array such as dot product or norm.
    engine : :obj:`str`, optional
        Engine used to store array (``numpy`` or ``cupy``)
    dtype : :obj:`str`, optional
        Type of elements in input array. Defaults to ``numpy.float64``.
    """

    def __init__(self, global_shape: Union[Tuple, Integral],
                 base_comm: Optional[Union[MPI.Comm, NcclCommunicator]] = MPI.COMM_WORLD,
                 partition: Partition = Partition.SCATTER, axis: int = 0,
                 local_shapes: Optional[List[Union[Tuple, Integral]]] = None,
                 mask: Optional[List[Integral]] = None,
                 engine: Optional[str] = "numpy",
                 dtype: Optional[DTypeLike] = np.float64):
        if isinstance(global_shape, Integral):
            global_shape = (global_shape,)
        if len(global_shape) <= axis:
            raise IndexError(f"Axis {axis} out of range for DistributedArray "
                             f"of shape {global_shape}")
        if partition not in Partition:
            raise ValueError(f"Should be either {Partition.BROADCAST}, "
                             f"{Partition.UNSAFE_BROADCAST} or {Partition.SCATTER}")
        if not isinstance(base_comm, MPI.Comm) and engine != "cupy":
            raise ValueError("NCCL Communicator only works with engine `cupy`")

        self.dtype = dtype
        self._global_shape = _value_or_sized_to_tuple(global_shape)
        self._base_comm = base_comm
        self._partition = partition
        self._axis = axis
        self._mask = mask
        self._sub_comm = base_comm if mask is None else subcomm_split(mask, base_comm)
        local_shapes = local_shapes if local_shapes is None else [_value_or_sized_to_tuple(local_shape) for local_shape in local_shapes]
        self._check_local_shapes(local_shapes)
        self._local_shape = local_shapes[self.rank] if local_shapes else local_split(global_shape, base_comm,
                                                                                     partition, axis)

        self._engine = engine
        self._local_array = get_module(engine).empty(shape=self.local_shape, dtype=self.dtype)

    def __getitem__(self, index):
        return self.local_array[index]

    def __setitem__(self, index, value):
        """Setter Method

        `Partition.SCATTER` - Local Arrays are assigned their
        unique values.

        `Partition.UNSAFE_SCATTER` - Local Arrays are assigned their
        unique values.

        `Partition.BROADCAST` - The value at rank-0 is broadcasted
        and is assigned to all the ranks.

        Parameters
        ----------
        index : :obj:`int` or :obj:`slice`
            Represents the index positions where a value needs to be assigned.
        value : :obj:`int` or :obj:`numpy.ndarray`
            Represents the value that will be assigned to the local array at
            the specified index positions.
        """

        if self.partition is Partition.BROADCAST:
            if isinstance(self.base_comm, MPI.Comm):
                self.local_array[index] = self.base_comm.bcast(value)
            else:
                nccl_bcast(self.base_comm, self.local_array, index, value)
        else:
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
    def mask(self):
        """Mask of the Distributed array

        Returns
        -------
        engine : :obj:`list`
        """
        return self._mask

    @property
    def engine(self):
        """Engine of the Distributed array

        Returns
        -------
        engine : :obj:`str`
        """
        return self._engine

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
        # cp.cuda.Device().id will give local rank
        # It works ok in the single-node multi-gpu environment.
        # But in multi-node environment, the function will break.
        # So we have to use MPI.COMM_WORLD() in both cases of base_comm (MPI and NCCL)
        return MPI.COMM_WORLD.Get_rank()

    @property
    def size(self):
        """Total number of processes
        Size of parallel environment

        Returns
        -------
        size : :obj:`int`
        """
        return MPI.COMM_WORLD.Get_size()

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

    @property
    def local_shapes(self):
        """Gather Local shapes from all ranks

        Returns
        -------
        local_shapes : :obj:`list`
        """
        if self.base_comm is MPI.COMM_WORLD:
            return self._allgather(self.local_shape)
        else:
            # gather tuple of shapes from every rank and copy from GPU to CPU
            all_tuples = self._allgather(self.local_shape).get()
            # NCCL returns the flat array that packs every tuple as 1-dimensional array
            # unpack each tuple from each rank
            tuple_len = len(self.local_shape)
            return [tuple(all_tuples[i : i + tuple_len]) for i in range(0, len(all_tuples), tuple_len)]

    @property
    def sub_comm(self):
        """MPI Sub-Communicator

        Returns
        -------
        sub_comm : :obj:`MPI.Comm`
        """
        return self._sub_comm

    def asarray(self):
        """Global view of the array

        Gather all the local arrays

        Returns
        -------
        final_array : :obj:`numpy.ndarray`
            Global Array gathered at all ranks
        """
        # Since the global array was replicated at all ranks
        if self.partition in [Partition.BROADCAST, Partition.UNSAFE_BROADCAST]:
            # Get only self.local_array.
            return self.local_array

        if isinstance(self.base_comm, MPI.Comm):
            # Gather all the local arrays and apply concatenation.
            final_array = self._allgather(self.local_array)
            return np.concatenate(final_array, axis=self.axis)
        else:
            return nccl_asarray(self.base_comm, self.local_array, self.local_shapes, self.axis)

    @classmethod
    def to_dist(cls, x: NDArray,
                base_comm: MPI.Comm = MPI.COMM_WORLD,
                partition: Partition = Partition.SCATTER,
                axis: int = 0,
                local_shapes: Optional[List[Tuple]] = None,
                mask: Optional[List[Integral]] = None):
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
        local_shapes : :obj:`list`, optional
            Local Shapes at each rank.
        mask : :obj:`list`, optional
            Mask defining subsets of ranks to consider when performing 'global'
            operations on the distributed array such as dot product or norm.

        Returns
        -------
        dist_array : :obj:`DistributedArray`
            Distributed Array of the Global Array
        """
        dist_array = DistributedArray(global_shape=x.shape,
                                      base_comm=base_comm,
                                      partition=partition,
                                      axis=axis,
                                      local_shapes=local_shapes,
                                      mask=mask,
                                      engine=get_module_name(get_array_module(x)),
                                      dtype=x.dtype)
        if partition in [Partition.BROADCAST, Partition.UNSAFE_BROADCAST]:
            dist_array[:] = x
        else:
            slices = [slice(None)] * x.ndim
            local_shapes = np.append([0], dist_array._allgather(
                dist_array.local_shape[axis]))
            sum_shapes = np.cumsum(local_shapes)
            slices[axis] = slice(sum_shapes[dist_array.rank],
                                 sum_shapes[dist_array.rank + 1], None)
            dist_array[:] = x[tuple(slices)]
        return dist_array

    def _check_local_shapes(self, local_shapes):
        """Check if the local shapes align with the global shape"""
        if local_shapes:
            if len(local_shapes) != self.size:
                raise ValueError(f"Length of local shapes is not equal to number of processes; "
                                 f"{len(local_shapes)} != {self.size}")
            # Check if local shape == global shape
            if self.partition in [Partition.BROADCAST, Partition.UNSAFE_BROADCAST] and local_shapes[self.rank] != self.global_shape:
                raise ValueError(f"Local shape is not equal to global shape at rank = {self.rank};"
                                 f"{local_shapes[self.rank]} != {self.global_shape}")
            elif self.partition is Partition.SCATTER:
                local_shape = local_shapes[self.rank]
                # Check if local shape sum up to global shape and other dimensions align with global shape
                if self._allreduce(local_shape[self.axis]) != self.global_shape[self.axis] or \
                        not np.array_equal(np.delete(local_shape, self.axis), np.delete(self.global_shape, self.axis)):
                    raise ValueError(f"Local shapes don't align with the global shape;"
                                     f"{local_shapes} != {self.global_shape}")

    def _check_partition_shape(self, dist_array):
        """Check Partition and Local Shape of the Array
        """
        if self.partition != dist_array.partition:
            raise ValueError("Partition of both the arrays must be same")
        if self.local_shape != dist_array.local_shape:
            raise ValueError(f"Local Array Shape Mismatch - "
                             f"{self.local_shape} != {dist_array.local_shape}")

    def _check_mask(self, dist_array):
        """Check mask of the Array
        """
        if not np.array_equal(self.mask, dist_array.mask):
            raise ValueError("Mask of both the arrays must be same")

    def _allreduce(self, send_buf, recv_buf=None, op: MPI.Op = MPI.SUM):
        """Allreduce operation
        """
        if isinstance(self.base_comm, MPI.Comm):
            if recv_buf is None:
                return self.base_comm.allreduce(send_buf, op)
            # For MIN and MAX which require recv_buf
            self.base_comm.Allreduce(send_buf, recv_buf, op)
            return recv_buf
        else:
            return nccl_allreduce(self.base_comm, send_buf, recv_buf, op)

    def _allreduce_subcomm(self, send_buf, recv_buf=None, op: MPI.Op = MPI.SUM):
        """Allreduce operation with subcommunicator
        """

        if isinstance(self.base_comm, MPI.Comm):
            if recv_buf is None:
                return self.sub_comm.allreduce(send_buf, op)
            # For MIN and MAX which require recv_buf
            self.sub_comm.Allreduce(send_buf, recv_buf, op)
            return recv_buf

        else:
            return nccl_allreduce(self.sub_comm, send_buf, recv_buf, op)

    def _allgather(self, send_buf, recv_buf=None):
        """Allgather operation
        """
        if isinstance(self.base_comm, MPI.Comm):
            if recv_buf is None:
                return self.base_comm.allgather(send_buf)
            self.base_comm.Allgather(send_buf, recv_buf)
            return recv_buf
        else:
            return nccl_allgather(self.base_comm, send_buf, recv_buf)

    def __neg__(self):
        arr = DistributedArray(global_shape=self.global_shape,
                               base_comm=self.base_comm,
                               partition=self.partition,
                               axis=self.axis,
                               local_shapes=self.local_shapes,
                               mask=self.mask,
                               engine=self.engine,
                               dtype=self.dtype)
        arr[:] = -self.local_array
        return arr

    def __add__(self, x):
        return self.add(x)

    def __iadd__(self, x):
        return self.iadd(x)

    def __sub__(self, x):
        return self.__add__(-x)

    def __isub__(self, x):
        return self.__iadd__(-x)

    def __mul__(self, x):
        return self.multiply(x)

    def __rmul__(self, x):
        return self.multiply(x)

    def add(self, dist_array):
        """Distributed Addition of arrays
        """
        self._check_partition_shape(dist_array)
        self._check_mask(dist_array)
        SumArray = DistributedArray(global_shape=self.global_shape,
                                    base_comm=self.base_comm,
                                    dtype=self.dtype,
                                    partition=self.partition,
                                    local_shapes=self.local_shapes,
                                    mask=self.mask,
                                    engine=self.engine,
                                    axis=self.axis)
        SumArray[:] = self.local_array + dist_array.local_array
        return SumArray

    def iadd(self, dist_array):
        """Distributed In-place Addition of arrays
        """
        self._check_partition_shape(dist_array)
        self._check_mask(dist_array)
        self[:] = self.local_array + dist_array.local_array
        return self

    def multiply(self, dist_array):
        """Distributed Element-wise multiplication
        """
        if isinstance(dist_array, DistributedArray):
            self._check_partition_shape(dist_array)
            self._check_mask(dist_array)

        ProductArray = DistributedArray(global_shape=self.global_shape,
                                        base_comm=self.base_comm,
                                        dtype=self.dtype,
                                        partition=self.partition,
                                        local_shapes=self.local_shapes,
                                        mask=self.mask,
                                        engine=self.engine,
                                        axis=self.axis)
        if isinstance(dist_array, DistributedArray):
            # multiply two DistributedArray
            ProductArray[:] = self.local_array * dist_array.local_array
        else:
            # multiply with scalar
            ProductArray[:] = self.local_array * dist_array
        return ProductArray

    def dot(self, dist_array):
        """Distributed Dot Product
        """
        self._check_partition_shape(dist_array)
        self._check_mask(dist_array)
        ncp = get_module(self.engine)
        # Convert to Partition.SCATTER if Partition.BROADCAST
        x = DistributedArray.to_dist(x=self.local_array) \
            if self.partition in [Partition.BROADCAST, Partition.UNSAFE_BROADCAST] else self
        y = DistributedArray.to_dist(x=dist_array.local_array) \
            if self.partition in [Partition.BROADCAST, Partition.UNSAFE_BROADCAST] else dist_array
        # Flatten the local arrays and calculate dot product
        return self._allreduce_subcomm(ncp.dot(x.local_array.flatten(), y.local_array.flatten()))

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
        ncp = get_module(self.engine)
        ord = 2 if ord is None else ord
        if local_array.ndim == 1:
            recv_buf = ncp.empty(shape=1, dtype=np.float64)
        else:
            global_shape = list(self.global_shape)
            global_shape[axis] = 1
            recv_buf = ncp.empty(shape=global_shape, dtype=ncp.float64)
        if ord in ['fro', 'nuc']:
            raise ValueError(f"norm-{ord} not possible for vectors")
        elif ord == 0:
            # Count non-zero then sum reduction
            recv_buf = self._allreduce_subcomm(ncp.count_nonzero(local_array, axis=axis).astype(ncp.float64))
        elif ord == ncp.inf:
            # Calculate max followed by max reduction
            recv_buf = self._allreduce_subcomm(ncp.max(ncp.abs(local_array), axis=axis).astype(ncp.float64),
                                               recv_buf, op=MPI.MAX)
            recv_buf = ncp.squeeze(recv_buf, axis=axis)
        elif ord == -ncp.inf:
            # Calculate min followed by min reduction
            recv_buf = self._allreduce_subcomm(ncp.min(ncp.abs(local_array), axis=axis).astype(ncp.float64),
                                               recv_buf, op=MPI.MIN)
            recv_buf = ncp.squeeze(recv_buf, axis=axis)

        else:
            recv_buf = self._allreduce_subcomm(ncp.sum(ncp.abs(ncp.float_power(local_array, ord)), axis=axis))
            recv_buf = ncp.power(recv_buf, 1.0 / ord)
        return recv_buf

    def zeros_like(self):
        """Creates a copy of the DistributedArray filled with zeros
        """
        arr = DistributedArray(global_shape=self.global_shape,
                               base_comm=self.base_comm,
                               partition=self.partition,
                               axis=self.axis,
                               local_shapes=self.local_shapes,
                               engine=self.engine,
                               dtype=self.dtype)
        arr[:] = 0.
        return arr

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
        # Convert to Partition.SCATTER if Partition.BROADCAST
        x = DistributedArray.to_dist(x=self.local_array) \
            if self.partition in [Partition.BROADCAST, Partition.UNSAFE_BROADCAST] else self
        if axis == -1:
            # Flatten the local arrays and calculate norm
            return x._compute_vector_norm(x.local_array.flatten(), axis=0, ord=ord)
        if axis != x.axis:
            raise NotImplementedError("Choose axis along the partition.")
        # Calculate vector norm along the axis
        return x._compute_vector_norm(x.local_array, axis=x.axis, ord=ord)

    def conj(self):
        """Distributed conj() method
        """
        conj = DistributedArray(global_shape=self.global_shape,
                                base_comm=self.base_comm,
                                partition=self.partition,
                                axis=self.axis,
                                local_shapes=self.local_shapes,
                                mask=self.mask,
                                engine=self.engine,
                                dtype=self.dtype)
        conj[:] = self.local_array.conj()
        return conj

    def copy(self):
        """Creates a copy of the DistributedArray
        """
        arr = DistributedArray(global_shape=self.global_shape,
                               base_comm=self.base_comm,
                               partition=self.partition,
                               axis=self.axis,
                               local_shapes=self.local_shapes,
                               mask=self.mask,
                               engine=self.engine,
                               dtype=self.dtype)
        arr[:] = self.local_array
        return arr

    def ravel(self, order: Optional[str] = "C"):
        """Return a flattened DistributedArray

        Parameters
        ----------
        order : :obj:`str`, optional
            Order in which array needs to be flattened.
            {'C','F', 'A', 'K'}

        Returns
        -------
        arr : :obj:`pylops_mpi.DistributedArray`
            Flattened 1-D DistributedArray
        """
        local_shapes = [(np.prod(local_shape, axis=-1), ) for local_shape in self.local_shapes]
        arr = DistributedArray(global_shape=np.prod(self.global_shape),
                               local_shapes=local_shapes,
                               mask=self.mask,
                               partition=self.partition,
                               engine=self.engine,
                               dtype=self.dtype)
        local_array = np.ravel(self.local_array, order=order)
        x = local_array.copy()
        arr[:] = x
        return arr

    def add_ghost_cells(self, cells_front: Optional[int] = None,
                        cells_back: Optional[int] = None):
        """Add ghost cells to the DistributedArray along the axis
        of partition at each rank.

        Parameters
        ----------
        cells_front : :obj:`int`, optional
            Number of cells to be added from the previous process
            to the start of the array at each rank. Defaults to ``None``.
        cells_back : :obj:`int`, optional
            Number of cells to be added from the next process
            to the back of the array at each rank. Defaults to ``None``.

        Returns
        -------
        ghosted_array : :obj:`numpy.ndarray`
            Ghosted Array

        """
        ghosted_array = self.local_array.copy()
        if cells_front is not None:
            total_cells_front = self._allgather(cells_front) + [0]
            # Read cells_front which needs to be sent to rank + 1(cells_front for rank + 1)
            cells_front = total_cells_front[self.rank + 1]
            if self.rank != 0:
                ghosted_array = np.concatenate([self.base_comm.recv(source=self.rank - 1, tag=1), ghosted_array],
                                               axis=self.axis)
            if self.rank != self.size - 1:
                if cells_front > self.local_shape[self.axis]:
                    raise ValueError(f"Local Shape at rank={self.rank} along axis={self.axis} "
                                     f"should be > {cells_front}: dim({self.axis}) "
                                     f"{self.local_shape[self.axis]} < {cells_front}; "
                                     f"to achieve this use NUM_PROCESSES <= "
                                     f"{max(1, self.global_shape[self.axis] // cells_front)}")
                self.base_comm.send(np.take(self.local_array, np.arange(-cells_front, 0), axis=self.axis),
                                    dest=self.rank + 1, tag=1)
        if cells_back is not None:
            total_cells_back = self._allgather(cells_back) + [0]
            # Read cells_back which needs to be sent to rank - 1(cells_back for rank - 1)
            cells_back = total_cells_back[self.rank - 1]
            if self.rank != 0:
                if cells_back > self.local_shape[self.axis]:
                    raise ValueError(f"Local Shape at rank={self.rank} along axis={self.axis} "
                                     f"should be > {cells_back}: dim({self.axis}) "
                                     f"{self.local_shape[self.axis]} < {cells_back}; "
                                     f"to achieve this use NUM_PROCESSES <= "
                                     f"{max(1, self.global_shape[self.axis] // cells_back)}")
                self.base_comm.send(np.take(self.local_array, np.arange(cells_back), axis=self.axis),
                                    dest=self.rank - 1, tag=0)
            if self.rank != self.size - 1:
                ghosted_array = np.append(ghosted_array, self.base_comm.recv(source=self.rank + 1, tag=0),
                                          axis=self.axis)
        return ghosted_array

    def __repr__(self):
        return f"<DistributedArray with global shape={self.global_shape}, " \
               f"local shape={self.local_shape}" \
               f", dtype={self.dtype}, " \
               f"processes={[i for i in range(self.size)]})> "


class StackedDistributedArray:
    r"""Stacked DistributedArrays

    Stack DistributedArray objects and power them with basic mathematical operations.
    This class allows one to work with a series of distributed arrays to avoid having to create
    a single distributed array with some special internal sorting.

    Parameters
    ----------
    distarrays : :obj:`list`
        List of :class:`pylops_mpi.DistributedArray` objects.
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        Base MPI Communicator.
        Defaults to ``mpi4py.MPI.COMM_WORLD``.
    """

    def __init__(self, distarrays: List, base_comm: MPI.Comm = MPI.COMM_WORLD):
        self.distarrays = distarrays
        self.narrays = len(distarrays)
        self.base_comm = base_comm
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.size = MPI.COMM_WORLD.Get_size()

    def __getitem__(self, index):
        return self.distarrays[index]

    def __setitem__(self, index, value):
        self.distarrays[index][:] = value

    def asarray(self):
        """Global view of the array

        Gather all the distributed arrays

        Returns
        -------
        final_array : :obj:`numpy.ndarray`
            Global Array gathered at all ranks

        """
        return np.hstack([distarr.asarray().ravel() for distarr in self.distarrays])

    def _check_stacked_size(self, stacked_array):
        """Check that arrays have consistent size

        """
        if self.narrays != stacked_array.narrays:
            raise ValueError("Stacked arrays must be composed the same number of of distributed arrays")
        for iarr in range(self.narrays):
            if self.distarrays[iarr].global_shape != stacked_array[iarr].global_shape:
                raise ValueError(f"Stacked arrays {iarr} have different global shape:"
                                 f"{self.distarrays[iarr].global_shape} / "
                                 f"{stacked_array[iarr].global_shape}")

    def __neg__(self):
        arr = self.copy()
        for iarr in range(self.narrays):
            arr[iarr][:] = -arr[iarr][:]
        return arr

    def __add__(self, x):
        return self.add(x)

    def __iadd__(self, x):
        return self.iadd(x)

    def __sub__(self, x):
        return self.__add__(-x)

    def __isub__(self, x):
        return self.__iadd__(-x)

    def __mul__(self, x):
        return self.multiply(x)

    def __rmul__(self, x):
        return self.multiply(x)

    def add(self, stacked_array):
        """Stacked Distributed Addition of arrays
        """
        self._check_stacked_size(stacked_array)
        SumArray = self.copy()
        for iarr in range(self.narrays):
            SumArray[iarr][:] = (self[iarr] + stacked_array[iarr])[:]
        return SumArray

    def iadd(self, stacked_array):
        """Stacked Distributed In-Place Addition of arrays
        """
        self._check_stacked_size(stacked_array)
        for iarr in range(self.narrays):
            self[iarr][:] = (self[iarr] + stacked_array[iarr])[:]
        return self

    def multiply(self, stacked_array):
        """Stacked Distributed Multiplication of arrays
        """
        if isinstance(stacked_array, StackedDistributedArray):
            self._check_stacked_size(stacked_array)
        ProductArray = self.copy()

        if isinstance(stacked_array, StackedDistributedArray):
            # multiply two DistributedArray
            for iarr in range(self.narrays):
                ProductArray[iarr][:] = (self[iarr] * stacked_array[iarr])[:]
        else:
            # multiply with scalar
            for iarr in range(self.narrays):
                ProductArray[iarr][:] = (self[iarr] * stacked_array)[:]
        return ProductArray

    def dot(self, stacked_array):
        """Dot Product of Stacked Distributed Arrays
        """
        self._check_stacked_size(stacked_array)
        dotprod = 0.
        for iarr in range(self.narrays):
            dotprod += self[iarr].dot(stacked_array[iarr])
        return dotprod

    def norm(self, ord: Optional[int] = None):
        """numpy.linalg.norm method on stacked Distributed arrays

        Parameters
        ----------
        ord : :obj:`int`, optional
            Order of the norm.
        """
        ncp = get_module(self.distarrays[0].engine)
        norms = ncp.array([distarray.norm(ord) for distarray in self.distarrays])
        ord = 2 if ord is None else ord
        if ord in ['fro', 'nuc']:
            raise ValueError(f"norm-{ord} not possible for vectors")
        elif ord == 0:
            # Count non-zero then sum reduction
            norm = ncp.sum(norms)
        elif ord == ncp.inf:
            # Calculate max followed by max reduction
            norm = ncp.max(norms)
        elif ord == -ncp.inf:
            # Calculate min followed by max reduction
            norm = ncp.min(norms)
        else:
            norm = ncp.power(ncp.sum(ncp.power(norms, ord)), 1. / ord)
        return norm

    def conj(self):
        """Distributed conj() method
        """
        ConjArray = StackedDistributedArray([distarray.conj() for distarray in self.distarrays])
        return ConjArray

    def copy(self):
        """Creates a copy of the DistributedArray
        """
        arr = StackedDistributedArray([distarray.copy() for distarray in self.distarrays])
        return arr

    def __repr__(self):
        repr_dist = "\n".join([distarray.__repr__() for distarray in self.distarrays])
        return f"<StackedDistributedArray with {self.narrays} distributed arrays: \n" + repr_dist
