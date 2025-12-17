__all__ = [
    "_nccl_sync",
    "initialize_nccl_comm",
    "nccl_split",
    "nccl_allgather",
    "nccl_allreduce",
    "nccl_bcast",
    "nccl_asarray",
    "nccl_send",
    "nccl_recv",
]

from enum import IntEnum
from mpi4py import MPI
import os
import cupy as cp
import cupy.cuda.nccl as nccl
from pylops_mpi.utils._common import _prepare_allgather_inputs, _unroll_allgather_recv

cupy_to_nccl_dtype = {
    "float32": nccl.NCCL_FLOAT32,
    "float64": nccl.NCCL_FLOAT64,
    "int32": nccl.NCCL_INT32,
    "int64": nccl.NCCL_INT64,
    "uint8": nccl.NCCL_UINT8,
    "int8": nccl.NCCL_INT8,
    "uint32": nccl.NCCL_UINT32,
    "uint64": nccl.NCCL_UINT64,
    # sending complex array as float with 2x size
    "complex64": nccl.NCCL_FLOAT32,
    "complex128": nccl.NCCL_FLOAT64,
}


class NcclOp(IntEnum):
    SUM = nccl.NCCL_SUM
    PROD = nccl.NCCL_PROD
    MAX = nccl.NCCL_MAX
    MIN = nccl.NCCL_MIN


def _nccl_buf_size(buf, count=None):
    """ Get an appropriate buffer size according to the dtype of buf

    Parameters
    ----------
    buf : :obj:`cupy.ndarray` or array-like
        The data buffer from the local GPU to be sent.

    count : :obj:`int`, optional
        Number of elements to send from `buf`, if not sending the every element in `buf`.
    Returns:
    -------
    :obj:`int`
        An appropriate number of elements to send from `send_buf` for NCCL communication.
    """
    if buf.dtype in ['complex64', 'complex128']:
        return 2 * count if count else 2 * buf.size
    else:
        return count if count else buf.size


def _nccl_sync():
    """A thin wrapper of CuPy's synchronization for protected import"""
    if cp.cuda.runtime.getDeviceCount() == 0:
        return
    cp.cuda.runtime.deviceSynchronize()


def mpi_op_to_nccl(mpi_op) -> NcclOp:
    """ Map MPI reduction operation to NCCL equivalent

    Parameters
    ----------
    mpi_op : :obj:`MPI.Op`
        A MPI reduction operation (e.g., MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN).

    Returns:
    -------
    NcclOp : :obj:`IntEnum`
        A corresponding NCCL reduction operation.
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
    """ Initialize NCCL world communicator for every GPU device

    Each GPU must be managed by exactly one MPI process.
    i.e. the number of MPI process launched must be equal to
    number of GPUs in communications

    Returns:
    -------
    nccl_comm : :obj:`cupy.cuda.nccl.NcclCommunicator`
        A corresponding NCCL communicator
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create a communicator for ranks on the same node
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    size_node = node_comm.Get_size()

    device_id = int(
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or (rank % size_node) % cp.cuda.runtime.getDeviceCount()
    )
    cp.cuda.Device(device_id).use()

    if rank == 0:
        with cp.cuda.Device(device_id):
            nccl_id_bytes = nccl.get_unique_id()
    else:
        nccl_id_bytes = None
    nccl_id_bytes = comm.bcast(nccl_id_bytes, root=0)

    nccl_comm = nccl.NcclCommunicator(size, nccl_id_bytes, rank)
    return nccl_comm


def nccl_split(mask) -> nccl.NcclCommunicator:
    """ NCCL-equivalent of MPI.Split()

    Splitting the communicator into multiple NCCL subcommunicators

    Parameters
    ----------
    mask : :obj:`list`
        Mask defining subsets of ranks to consider when performing 'global'
        operations on the distributed array such as dot product or norm.

    Returns:
    -------
    sub_comm : :obj:`cupy.cuda.nccl.NcclCommunicator`
        Subcommunicator according to mask
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


def nccl_allgather(nccl_comm, send_buf, recv_buf=None) -> cp.ndarray:
    """ NCCL equivalent of MPI_Allgather. Gathers data from all GPUs
    and distributes the concatenated result to all participants.

    Parameters
    ----------
    nccl_comm : :obj:`cupy.cuda.nccl.NcclCommunicator`
        The NCCL communicator over which data will be gathered.
    send_buf : :obj:`cupy.ndarray` or array-like
        The data buffer from the local GPU to be sent.
    recv_buf : :obj:`cupy.ndarray`, optional
        The buffer to receive data from all GPUs. If None, a new
        buffer will be allocated with the appropriate shape.

    Returns
    -------
    recv_buf : :obj:`cupy.ndarray`
        A buffer containing the gathered data from all GPUs.
    """
    send_buf = (
        send_buf if isinstance(send_buf, cp.ndarray) else cp.asarray(send_buf)
    )
    if recv_buf is None:
        recv_buf = cp.zeros(
            nccl_comm.size() * send_buf.size,
            dtype=send_buf.dtype,
        )
    nccl_comm.allGather(
        send_buf.data.ptr,
        recv_buf.data.ptr,
        _nccl_buf_size(send_buf),
        cupy_to_nccl_dtype[str(send_buf.dtype)],
        cp.cuda.Stream.null.ptr,
    )
    return recv_buf


def nccl_allreduce(nccl_comm, send_buf, recv_buf=None, op: MPI.Op = MPI.SUM) -> cp.ndarray:
    """ NCCL equivalent of MPI_Allreduce. Applies a reduction operation
    (e.g., sum, max) across all GPUs and distributes the result.

    Parameters
    ----------
    nccl_comm : :obj:`cupy.cuda.nccl.NcclCommunicator`
        The NCCL communicator used for collective communication.
    send_buf : :obj:`cupy.ndarray` or array-like
        The data buffer from the local GPU to be reduced.
    recv_buf : :obj:`cupy.ndarray`, optional
        The buffer to store the result of the reduction. If None,
        a new buffer will be allocated with the appropriate shape.
    op : :obj:mpi4py.MPI.Op, optional
        The reduction operation to apply. Defaults to MPI.SUM.

    Returns
    -------
    recv_buf : :obj:`cupy.ndarray`
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
        _nccl_buf_size(send_buf),
        cupy_to_nccl_dtype[str(send_buf.dtype)],
        mpi_op_to_nccl(op),
        cp.cuda.Stream.null.ptr,
    )
    return recv_buf


def nccl_bcast(nccl_comm, send_buf, root: int = 0) -> None:
    """NCCL equivalent of MPI_Bcast for an array buffer.

    Notes
    -----
    Any root-only assignment (e.g., setting a value prior to broadcast) must be
    done outside this function.
    """
    send_buf = send_buf if isinstance(send_buf, cp.ndarray) else cp.asarray(send_buf)
    nccl_comm.bcast(
        send_buf.data.ptr,
        _nccl_buf_size(send_buf),
        cupy_to_nccl_dtype[str(send_buf.dtype)],
        root,
        cp.cuda.Stream.null.ptr,
    )


def nccl_asarray(nccl_comm, local_array, local_shapes, axis) -> cp.ndarray:
    """Global view of the array

    Gather all local GPU arrays into a single global array via NCCL all-gather.

    Parameters
    ----------
    nccl_comm : :obj:`cupy.cuda.nccl.NcclCommunicator`
        The NCCL communicator used for collective communication.
    local_array : :obj:`cupy.ndarray`
        The local array on the current GPU.
    local_shapes : :obj:`list`
        A list of shapes for each GPU local array (used to trim padding).
    axis : :obj:`int`
        The axis along which to concatenate the gathered arrays.

    Returns
    -------
    final_array : :obj:`cupy.ndarray`
        Global array gathered from all GPUs and concatenated along `axis`.
    """

    send_buf, recv_buf = _prepare_allgather_inputs(local_array, local_shapes, engine="cupy")
    nccl_allgather(nccl_comm, send_buf, recv_buf)
    chunks = _unroll_allgather_recv(recv_buf, send_buf.shape, local_shapes)

    # combine back to single global array
    return cp.concatenate(chunks, axis=axis)


def nccl_send(nccl_comm, send_buf, dest, count):
    """NCCL equivalent of MPI_Send. Sends a specified number of elements
    from the buffer to a destination GPU device.

    Parameters
    ----------
    nccl_comm : :obj:`cupy.cuda.nccl.NcclCommunicator`
        The NCCL communicator used for point-to-point communication.
    send_buf : :obj:`cupy.ndarray`
        The array containing data to send.
    dest: :obj:`int`
        The rank of the destination GPU device.
    count : :obj:`int`
        Number of elements to send from `send_buf`.
    """
    nccl_comm.send(send_buf.data.ptr,
                   _nccl_buf_size(send_buf, count),
                   cupy_to_nccl_dtype[str(send_buf.dtype)],
                   dest,
                   cp.cuda.Stream.null.ptr
                   )


def nccl_recv(nccl_comm, recv_buf, source, count=None):
    """NCCL equivalent of MPI_Recv. Receives data from a source GPU device
    into the given buffer.

    Parameters
    ----------
    nccl_comm : :obj:`cupy.cuda.nccl.NcclCommunicator`
        The NCCL communicator used for point-to-point communication.
    recv_buf : :obj:`cupy.ndarray`
        The array to store the received data.
    source : :obj:`int`
        The rank of the source GPU device.
    count : :obj:`int`, optional
        Number of elements to receive.
    """
    nccl_comm.recv(recv_buf.data.ptr,
                   _nccl_buf_size(recv_buf, count),
                   cupy_to_nccl_dtype[str(recv_buf.dtype)],
                   source,
                   cp.cuda.Stream.null.ptr
                   )
