__all__ = [
    "_unroll_allgather_recv"
]

import numpy as np


def _unroll_allgather_recv(recv_buf, chunk_shape, send_buf_shapes, displs=None) -> list:
    r"""Unroll recv_buf after Buffered Allgather (MPI and NCCL)

    Depending on the provided parameters, the function:
    - uses ``displs`` and element counts to extract variable-sized chunks.
    - removes padding and reshapes each chunk using ``chunk_shape``.

    Each rank may send an array with a different shape, so the return type is a list of arrays
    instead of a concatenated array.

    Parameters
    ----------
    recv_buf: :obj:`cupy.ndarray` or array-like
        The data buffer returned from nccl_allgather call
    send_buf_shapes: :obj:`list`
        A list of original shapes of each rank's send_buf before any padding.
    chunk_shape : tuple
        Shape of each gathered chunk in recv_buf. This must match the shape
        used to construct the gathered buffer: use the padded send buffer shape
        when padding is required (e.g., nccl_allgather with padding), or the original send buffer
        shape when no padding is used.
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
        recvcounts = [int(np.prod(shape)) for shape in send_buf_shapes]
        # Slice recv_buf using displacements and then reconstruct the original-shaped chunk.
        return [
            recv_buf[displs[i]:displs[i] + recvcounts[i]].reshape(send_buf_shapes[i])
            for i in range(ndev)
        ]
    else:
        chunk_size = np.prod(chunk_shape)
        chunks = [
            recv_buf[i * chunk_size:(i + 1) * chunk_size]
            for i in range(ndev)
        ]
        # Remove padding from each array: the padded value may appear somewhere
        # in the middle of the flat array and thus the reshape and slicing for each dimension is required
        for i in range(ndev):
            slicing = tuple(slice(0, end) for end in send_buf_shapes[i])
            chunks[i] = chunks[i].reshape(chunk_shape)[slicing]
        return chunks
