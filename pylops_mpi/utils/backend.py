__all__ = [
    "initialize_nccl_comm",
    "get_cupy",
    "nccl_split",
    "nccl_allgather",
    "nccl_allreduce",
    "nccl_bcast",
    "nccl_asarray"
]

from types import ModuleType

from enum import IntEnum
from mpi4py import MPI
from pylops.utils.deps import cupy_enabled
from pylops_mpi.utils import deps
import os

cupy_to_nccl_dtype = {}
if cupy_enabled:
    import cupy as cp
else:
    cp = None

# TODO: nccl must always come with CuPy (this may not be necessary check ?)
if deps.nccl_enabled and not (cp is None):
    import cupy.cuda.nccl as nccl

    cupy_to_nccl_dtype.update({
        "float32": nccl.NCCL_FLOAT32,
        "float64": nccl.NCCL_FLOAT64,
        "int32": nccl.NCCL_INT32,
        "int64": nccl.NCCL_INT64,
        "uint8": nccl.NCCL_UINT8,
        "int8": nccl.NCCL_INT8,
        "uint32": nccl.NCCL_UINT32,
        "uint64": nccl.NCCL_UINT64,
    })

    class NcclOp(IntEnum):
        SUM = nccl.NCCL_SUM
        PROD = nccl.NCCL_PROD
        MAX = nccl.NCCL_MAX
        MIN = nccl.NCCL_MIN

    def mpi_op_to_nccl(mpi_op) -> NcclOp:
        """ Map MPI reduction operation to NCCL equivalent
        Parameters
        ----------
        mpi_op (MPI.Op): An MPI reduction operation (e.g., MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN).

        Returns:
        -------
        NcclOp: The corresponding NCCL reduction operation.
        """
        if mpi_op is MPI.SUM:
            return NcclOp.SUM
        elif mpi_op is MPI.PROD:
            return NcclOp.PROD
        elif mpi_op is MPI.MAX:
            return NcclOp.MAX
        elif mpi_op is MPI.MIN:
            return NcclOp.MIN
        else:
            raise ValueError(f"Unsupported MPI.Op for NCCL: {mpi_op}")

    def initialize_nccl_comm() -> nccl.NcclCommunicator:
        """ Initilize NCCL world communicator for every GPU device
        Each GPU must be managed by exactly one MPI process.
        i.e. the number of MPI process launched must be equal to
        number of GPUs in communications

        Returns:
        -------
        cupy.cuda.nccl.NcclCommunicator: The corresponding NCCL communicator
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        local_rank = int(
            os.environ.get("SLURM_LOCALID")
            or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
            or rank % cp.cuda.runtime.getDeviceCount()
        )
        device_id = local_rank
        cp.cuda.Device(device_id).use()

        if rank == 0:
            nccl_id = nccl.get_unique_id()
        else:
            nccl_id = None
        nccl_id = comm.bcast(nccl_id, root=0)

        nccl_comm = nccl.NcclCommunicator(size, nccl_id, rank)
        return nccl_comm

    def nccl_split(mask):
        """ NCCL-equivalent of MPI.Split(). Splitting the communicator
        into multiple NCCL subcommunicators

        Parameters
        ----------
        mask : :obj:`list`
            Mask defining subsets of ranks to consider when performing 'global'
            operations on the distributed array such as dot product or norm.

        Returns:
        -------
            cupy.cuda.nccl.NcclCommunicator: a NCCL subcommunicator according to mask
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        sub_comm = comm.Split(color=mask[rank], key=rank)

        sub_rank = sub_comm.Get_rank()
        sub_size = sub_comm.Get_size()

        if sub_rank == 0:
            nccl_id_bytes = nccl.get_unique_id()
        else:
            nccl_id_bytes = None
        nccl_id_bytes = sub_comm.bcast(nccl_id_bytes, root=0)
        sub_comm = nccl.NcclCommunicator(sub_size, nccl_id_bytes, sub_rank)

        return sub_comm

    def nccl_allgather(nccl_comm, send_buf, recv_buf=None):
        """ NCCL equivalent of MPI_Allgather. Gathers data from all GPUs
        and distributes the concatenated result to all participants.

        Parameters
        ----------
        nccl_comm : cupy.cuda.nccl.NcclCommunicator
            The NCCL communicator over which data will be gathered.
        send_buf : :obj:`cupy.ndarray` or array-like
            The data buffer from the local GPU to be sent.
        recv_buf : :obj:`cupy.ndarray`, optional
            The buffer to receive data from all GPUs. If None, a new
            buffer will be allocated with the appropriate shape.

        Returns
        -------
        cupy.ndarray
            A buffer containing the gathered data from all GPUs.
        """
        send_buf = (
            send_buf if isinstance(send_buf, cp.ndarray) else cp.asarray(send_buf)
        )
        if recv_buf is None:
            recv_buf = cp.zeros(
                MPI.COMM_WORLD.Get_size() * send_buf.size,
                dtype=send_buf.dtype,
            )
        nccl_comm.allGather(
            send_buf.data.ptr,
            recv_buf.data.ptr,
            send_buf.size,
            cupy_to_nccl_dtype[str(send_buf.dtype)],
            cp.cuda.Stream.null.ptr,
        )
        return recv_buf

    def nccl_allreduce(nccl_comm, send_buf, recv_buf=None, op: MPI.Op = MPI.SUM):
        """ NCCL equivalent of MPI_Allreduce. Applies a reduction operation
        (e.g., sum, max) across all GPUs and distributes the result.

        Parameters
        ----------
        nccl_comm : cupy.cuda.nccl.NcclCommunicator
            The NCCL communicator used for collective communication.
        send_buf : :obj:`cupy.ndarray` or array-like
            The data buffer from the local GPU to be reduced.
        recv_buf : :obj:`cupy.ndarray`, optional
            The buffer to store the result of the reduction. If None,
            a new buffer will be allocated with the appropriate shape.
        op : mpi4py.MPI.Op, optional
            The reduction operation to apply. Defaults to MPI.SUM.

        Returns
        -------
        cupy.ndarray
            A buffer containing the result of the reduction, broadcasted
            to all GPUs.
        """
        send_buf = (
            send_buf if isinstance(send_buf, cp.ndarray) else cp.asarray(send_buf)
        )
        if recv_buf is None:
            recv_buf = cp.zeros(send_buf.size, dtype=send_buf.dtype)

        nccl_comm.allReduce(
            send_buf.data.ptr,
            recv_buf.data.ptr,
            send_buf.size,
            cupy_to_nccl_dtype[str(send_buf.dtype)],
            mpi_op_to_nccl(op),
            cp.cuda.Stream.null.ptr,
        )
        return recv_buf

    def nccl_bcast(nccl_comm, local_array, index, value):
        """ NCCL equivalent of MPI_Bcast. Broadcasts a single value at the given index
        from the root GPU (rank 0) to all other GPUs.

        Parameters
        ----------
        nccl_comm : cupy.cuda.nccl.NcclCommunicator
            The NCCL communicator used for collective communication.
        local_array : cupy.ndarray
            The local array on each GPU. The value at `index` will be broadcasted.
        index : int
            The index in the array to be broadcasted.
        value : scalar
            The value to broadcast (only used by the root GPU, rank 0).

        Returns
        -------
        None
        """
        if nccl_comm.rank_id() == 0:
            local_array[index] = value
        nccl_comm.bcast(
            local_array[index].data.ptr,
            local_array[index].size,
            cupy_to_nccl_dtype[str(local_array[index].dtype)],
            0,
            cp.cuda.Stream.null.ptr,
        )

    def nccl_asarray(nccl_comm, local_array, local_shapes, axis):
        """Global view of the array

        Gather all local GPU arrays into a single global array via NCCL all-gather.

        Parameters
        ----------
        nccl_comm : cupy.cuda.nccl.NcclCommunicator
            The NCCL communicator used for collective communication.
        local_array : :obj:`cupy.ndarray`
            The local array on the current GPU.
        local_shapes : :obj:`list`
            A list of shapes for each GPU’s local array (used to trim padding).
        axis : int
            The axis along which to concatenate the gathered arrays.

        Returns
        -------
        final_array : :obj:`cupy.ndarray`
            Global array gathered from all GPUs and concatenated along `axis`.
        """
        sizes_each_dim = list(zip(*local_shapes))
        # NCCL allGather requires the send_buf to have the same size for every device
        send_shape = tuple(map(max, sizes_each_dim))
        pad_size = [
            (0, send_shape[i] - local_array.shape[i])
            for i in range(len(send_shape))
        ]

        send_buf = cp.pad(
            local_array, pad_size, mode="constant", constant_values=0
        )

        # NCCL recommends to use one MPI Process per GPU
        ndev = MPI.COMM_WORLD.Get_size()
        recv_buf = cp.zeros(ndev * send_buf.size, dtype=send_buf.dtype)
        # self._allgather(send_buf, recv_buf)
        nccl_allgather(nccl_comm, send_buf, recv_buf)

        chunk_size = cp.prod(cp.asarray(send_shape))
        chunks = [
            recv_buf[i * chunk_size:(i + 1) * chunk_size] for i in range(ndev)
        ]

        for i in range(ndev):
            slicing = tuple(slice(0, end) for end in local_shapes[i])
            chunks[i] = chunks[i].reshape(send_shape)[slicing]
        final_array = cp.concatenate([chunks[i] for i in range(ndev)], axis=axis)
        return final_array


def get_cupy() -> ModuleType:
    return cp
