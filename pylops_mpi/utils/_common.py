__all__ = [
    "_float_scalar",
    "_prepare_allgather_inputs",
    "_unroll_allgather_recv",
]

import numpy as np
from pylops.utils.backend import get_module


def _float_scalar(value) -> float:
    """Convert scalar or one-element ndarray/cupy.ndarray values to float."""
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


# TODO: return type annotation for both cupy and numpy
def _prepare_allgather_inputs(send_buf, send_buf_shapes, engine):
    r"""Prepare send_buf and recv_buf for buffered allgather

    Buffered Allgather (MPI and NCCL) requires the sending buffer to have the
    same size for every rank/device.
    Therefore, padding is required when the array is not evenly partitioned across
    all the ranks. The padding is applied such that each dimension of the sending buffers
    is equal to the max size of that dimension across all ranks.

    Similarly, each receiver buffer (recv_buf) is created with size equal to
    :math:`n_rank \cdot send_buf.size`

    Parameters
    ----------
    send_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray` or array-like
        The data buffer from the local rank/device to be sent for allgather.
    send_buf_shapes : :obj:`list`
        A list of shapes for each rank/device send_buf (used to calculate padding size)
    engine : :obj:`str`
        Engine used to store array (``numpy`` or ``cupy``)

    Returns
    -------
    send_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        A buffer containing the data and padded elements to be sent by this rank.
    recv_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        An empty, padded buffer to gather data from all ranks.
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


def _unroll_allgather_recv(recv_buf, padded_send_buf_shape, send_buf_shapes, displs=None) -> list:
    r"""Unroll recv_buf after Buffered Allgather (MPI and NCCL)

    Depending on the provided parameters, the function:
    - uses ``displs`` and element counts to extract variable-sized chunks.
    - removes padding and reshapes each chunk using ``padded_send_buf_shape``.

    Each rank may send an array with a different shape, so the return type is a list of arrays
    instead of a concatenated array.

    Parameters
    ----------
    recv_buf: :obj:`cupy.ndarray` or array-like
        The data buffer returned from the allgather call
    send_buf_shapes: :obj:`list`
        A list of original shapes of each rank's send_buf before any padding.
    padded_send_buf_shape : tuple
        Shape of each rank's data as stored in ``recv_buf``. This should match
        the layout used during allgather: use the padded send buffer shape when
        padding is applied (e.g., NCCL), or the original send buffer shape when
        no padding is used.
    displs : list, optional
        Starting offsets in recv_buf for each rank's data, used when chunks have
        variable sizes (e.g., mpi_allgather with displacements).

    Returns
    -------
    chunks : list of ndarray
        List of arrays (NumPy or CuPy, depending on ``engine``), one per rank,
        reconstructed to their original shapes with any padding removed.
    """
    ndev = len(send_buf_shapes)
    if displs is not None:
        return [
            recv_buf[displs[i]:displs[i] + int(np.prod(shape))].reshape(shape)
            for i, shape in enumerate(send_buf_shapes)
        ]

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
