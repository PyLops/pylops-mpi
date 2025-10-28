__all__ = [
    "_prepare_allgather_inputs",
    "_unroll_allgather_recv"
]


import numpy as np
from pylops.utils.backend import get_module


# TODO: return type annotation for both cupy and numpy
def _prepare_allgather_inputs(send_buf, send_buf_shapes, engine):
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


def _unroll_allgather_recv(recv_buf, padded_send_buf_shape, send_buf_shapes) -> list:
    r"""Unrolll recv_buf after Buffered Allgather (MPI and NCCL)

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
