__all__ = [
    "_prepare_allgather_inputs_nccl",
    "_prepare_allgather_inputs_mpi",
    "_unroll_allgather_recv"
]


import numpy as np
from pylops.utils.backend import get_module


# TODO: return type annotation for both cupy and numpy
def _prepare_allgather_inputs_nccl(send_buf, send_buf_shapes, engine):
    r""" Prepare send_buf and recv_buf for NCCL allgather (nccl_allgather)

    Buffered Allgather (MPI and NCCL) requires the sending buffer to have the same size for every device.
    Therefore, padding is required when the array is not evenly partitioned across
    all the ranks. The padding is applied such that the each dimension of the sending buffers
    is equal to the max size of that dimension across all ranks.

    Similarly, each receiver buffer (recv_buf) is created with size equal to :math:n_rank \cdot send_buf.size

    Parameters
    ----------
    send_buf : :obj: `numpy.ndarray` or `cupy.ndarray` or array-like
        The data buffer from the local GPU to be sent for allgather.
    send_buf_shapes: :obj:`list`
        A list of shapes for each GPU send_buf (used to calculate padding size)
    engine : :obj:`str`
        Engine used to store array (``numpy`` or ``cupy``)

    Returns
    -------
    send_buf: :obj:`cupy.ndarray`
        A buffer containing the data and padded elements to be sent by this rank.
    recv_buf : :obj:`cupy.ndarray`
        An empty, padded buffer to gather data from all GPUs.
    """
    ncp = get_module(engine)
    sizes_each_dim = list(zip(*send_buf_shapes))
    send_shape = tuple(map(max, sizes_each_dim))
    pad_size = [
        (0, s_shape - l_shape) for s_shape, l_shape in zip(send_shape, send_buf.shape)
    ]

    send_buf = ncp.pad(
        send_buf, pad_size, mode="constant", constant_values=0
    )

    ndev = len(send_buf_shapes)
    recv_buf = ncp.zeros(ndev * send_buf.size, dtype=send_buf.dtype)

    return send_buf, recv_buf


def _prepare_allgather_inputs_mpi(send_buf, send_buf_shapes, recvcounts, engine):
    r"""
    Prepare send_buf and recv_buf for MPI allgather (mpi_allgather)

    Parameters
    ----------
    send_buf : :obj: `numpy.ndarray` or `cupy.ndarray` or array-like
        The data buffer to be sent for allgather.
    send_buf_shapes: :obj:`list`
        A list of shapes for each send_buf (used to calculate padding size)
    recvcounts: :obj:`list`
        The element counts per rank in mpi_allgather
    engine : :obj:`str`
        Engine used to store array (``numpy`` or ``cupy``)

    Returns
    -------
    send_buf: :obj: `numpy.ndarray` or `cupy.ndarray` or array-like
        A buffer containing the data and padded elements to be sent by this rank.
    recv_buf : :obj: `numpy.ndarray` or `cupy.ndarray` or array-like
        A buffer to gather data from all ranks.
    displs : list, optional
        The starting offsets in recv_buf where data from each rank in mpi_allgather
    """
    ncp = get_module(engine)
    recv_buf = ncp.zeros(sum(recvcounts), dtype=send_buf.dtype)
    if len(set(send_buf_shapes)) == 1:
        displs = None
    else:
        displs = [0]
        for i in range(1, len(recvcounts)):
            displs.append(displs[i - 1] + recvcounts[i - 1])
    return ncp.ascontiguousarray(send_buf), recv_buf, displs


def _unroll_allgather_recv(recv_buf, send_buf_shapes, padded_send_buf_shape=None,
                           recvcounts=None, displs=None, engine="numpy") -> list:
    r"""Unroll recv_buf after Buffered Allgather (MPI and NCCL)

    Remove the padded elements in recv_buff, extract an individual array from each device and return them as a list of arrays
    Each GPU may send array with a different shape, so the return type has to be a list of array
    instead of the concatenated array.

    Parameters
    ----------
    recv_buf: :obj:`cupy.ndarray` or array-like
        The data buffer returned from nccl_allgather call
    send_buf_shapes: :obj:`list`
        A list of original shapes for each GPU send_buf prior to padding
    padded_send_buf_shape : tuple, optional
        The size of send_buf after padding used in nccl_allgather
    recvcounts : list, optional
        The element counts per rank in mpi_allgather
    displs : list, optional
        The starting offsets in recv_buf where data from each rank in mpi_allgather
    engine : :obj:`str`
        Engine used to store array (``numpy`` or ``cupy``)
    Returns
    -------
    chunks: :obj:`list`
        A list of `cupy.ndarray` from each GPU with the padded element removed
    """
    ndev = len(send_buf_shapes)
    if padded_send_buf_shape is not None:
        chunk_size = np.prod(padded_send_buf_shape)
        chunks = [
            recv_buf[i * chunk_size:(i + 1) * chunk_size]
            for i in range(ndev)
        ]
        for i in range(ndev):
            slicing = tuple(slice(0, end) for end in send_buf_shapes[i])
            chunks[i] = chunks[i].reshape(padded_send_buf_shape)[slicing]
        return chunks
    if displs is not None:
        return [
            recv_buf[displs[i]:displs[i] + recvcounts[i]].reshape(send_buf_shapes[i])
            for i in range(ndev)
        ]
    ncp = get_module(engine)
    chunks = ncp.split(recv_buf, ndev)
    return [chunk.reshape(send_buf_shapes[0]) for chunk in chunks]
