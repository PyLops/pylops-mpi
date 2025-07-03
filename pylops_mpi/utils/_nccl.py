__all__ = [
    "_prepare_nccl_allgather_inputs",
    "_unroll_nccl_allgather_recv",
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
from typing import Tuple
from mpi4py import MPI
import os
import numpy as np
import cupy as cp
import cupy.cuda.nccl as nccl


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


def _prepare_nccl_allgather_inputs(send_buf, send_buf_shapes) -> Tuple[cp.ndarray, cp.ndarray]:
    r""" Prepare send_buf and recv_buf for NCCL allgather (nccl_allgather)

    NCCL's allGather requires the sending buffer to have the same size for every device.
    Therefore, padding is required when the array is not evenly partitioned across
    all the ranks. The padding is applied such that the each dimension of the sending buffers
    is equal to the max size of that dimension across all ranks.

    Similarly, each receiver buffer (recv_buf) is created with size equal to :math:n_rank \cdot send_buf.size

    Parameters
    ----------
    send_buf : :obj:`cupy.ndarray` or array-like
        The data buffer from the local GPU to be sent for allgather.
    send_buf_shapes: :obj:`list`
        A list of shapes for each GPU send_buf (used to calculate padding size)

    Returns
    -------
    send_buf: :obj:`cupy.ndarray`
        A buffer containing the data and padded elements to be sent by this rank.
    recv_buf : :obj:`cupy.ndarray`
        An empty, padded buffer to gather data from all GPUs.
    """
    sizes_each_dim = list(zip(*send_buf_shapes))
    send_shape = tuple(map(max, sizes_each_dim))
    pad_size = [
        (0, s_shape - l_shape) for s_shape, l_shape in zip(send_shape, send_buf.shape)
    ]

    send_buf = cp.pad(
        send_buf, pad_size, mode="constant", constant_values=0
    )

    # NCCL recommends to use one MPI Process per GPU and so size of receiving buffer can be inferred
    ndev = len(send_buf_shapes)
    recv_buf = cp.zeros(ndev * send_buf.size, dtype=send_buf.dtype)

    return send_buf, recv_buf


def _unroll_nccl_allgather_recv(recv_buf, padded_send_buf_shape, send_buf_shapes) -> list:
    """Unrolll recv_buf after NCCL allgather (nccl_allgather)

    Remove the padded elements in recv_buff, extract an individual array from each device and return them as a list of arrays
    Each GPU may send array with a different shape, so the return type has to be a list of array
    instead of the concatenated array.

    Parameters
    ----------
    recv_buf: :obj:`cupy.ndarray` or array-like
        The data buffer returned from nccl_allgather call
    padded_send_buf_shape: :obj:`tuple`:int
        The size of send_buf after padding used in nccl_allgather
    send_buf_shapes: :obj:`list`
        A list of original shapes for each GPU send_buf prior to padding

    Returns
    -------
    chunks: :obj:`list`
        A list of `cupy.ndarray` from each GPU with the padded element removed
    """

    ndev = len(send_buf_shapes)
    # extract an individual array from each device
    chunk_size = np.prod(padded_send_buf_shape)
    chunks = [
        recv_buf[i * chunk_size:(i + 1) * chunk_size] for i in range(ndev)
    ]

    # Remove padding from each array: the padded value may appear somewhere
    # in the middle of the flat array and thus the reshape and slicing for each dimension is required
    for i in range(ndev):
        slicing = tuple(slice(0, end) for end in send_buf_shapes[i])
        chunks[i] = chunks[i].reshape(padded_send_buf_shape)[slicing]

    return chunks


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
    device_id = int(
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or rank % cp.cuda.runtime.getDeviceCount()
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


def nccl_bcast(nccl_comm, local_array, index, value) -> None:
    """ NCCL equivalent of MPI_Bcast. Broadcasts a single value at the given index
    from the root GPU (rank 0) to all other GPUs.

    Parameters
    ----------
    nccl_comm : :obj:`cupy.cuda.nccl.NcclCommunicator`
        The NCCL communicator used for collective communication.
    local_array : :obj:`cupy.ndarray`
        The local array on each GPU. The value at `index` will be broadcasted.
    index : :obj:`int`
        The index in the array to be broadcasted.
    value : :obj:`scalar`
        The value to broadcast (only used by the root GPU, rank 0).
    """
    if nccl_comm.rank_id() == 0:
        local_array[index] = value
    nccl_comm.bcast(
        local_array[index].data.ptr,
        _nccl_buf_size(local_array[index]),
        cupy_to_nccl_dtype[str(local_array[index].dtype)],
        0,
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

    send_buf, recv_buf = _prepare_nccl_allgather_inputs(local_array, local_shapes)
    nccl_allgather(nccl_comm, send_buf, recv_buf)
    chunks = _unroll_nccl_allgather_recv(recv_buf, send_buf.shape, local_shapes)

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
