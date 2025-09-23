__all__ = [
    "mpi_allgather",
    "mpi_allreduce",
    "mpi_bcast",
    # "mpi_asarray",
    "mpi_send",
    "mpi_recv",
]

from typing import Optional

import numpy as np
from mpi4py import MPI
from pylops.utils.backend import get_module
from pylops_mpi.utils import deps
from pylops_mpi.utils._common import _prepare_allgather_inputs, _unroll_allgather_recv


def mpi_allgather(base_comm: MPI.Comm,
                  send_buf, recv_buf=None,
                  engine: Optional[str] = "numpy") -> np.ndarray:

    if deps.cuda_aware_mpi_enabled or engine == "numpy":
        send_shapes = base_comm.allgather(send_buf.shape)
        (padded_send, padded_recv) = _prepare_allgather_inputs(send_buf, send_shapes, engine=engine)
        recv_buffer_to_use = recv_buf if recv_buf else padded_recv
        base_comm.Allgather(padded_send, recv_buffer_to_use)
        return _unroll_allgather_recv(recv_buffer_to_use, padded_send.shape, send_shapes)

    else:
        # CuPy with non-CUDA-aware MPI
        if recv_buf is None:
            return base_comm.allgather(send_buf)
        base_comm.Allgather(send_buf, recv_buf)
        return recv_buf


def mpi_allreduce(base_comm: MPI.Comm,
                  send_buf, recv_buf=None,
                  engine: Optional[str] = "numpy",
                  op: MPI.Op = MPI.SUM) -> np.ndarray:
    """MPI_Allreduce/allreduce

    Dispatch allreduce routine based on type of input and availability of
    CUDA-Aware MPI

    Parameters
    ----------
    base_comm : :obj:`MPI.Comm`
        Base MPI Communicator.
    send_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        The data buffer from the local GPU to be reduced.
    recv_buf : :obj:`cupy.ndarray`, optional
        The buffer to store the result of the reduction. If None,
        a new buffer will be allocated with the appropriate shape.
    engine : :obj:`str`, optional
        Engine used to store array (``numpy`` or ``cupy``)
    op : :obj:mpi4py.MPI.Op, optional
        The reduction operation to apply. Defaults to MPI.SUM.

    Returns
    -------
    recv_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        A buffer containing the result of the reduction, broadcasted
        to all GPUs.

    """
    if deps.cuda_aware_mpi_enabled or engine == "numpy":
        ncp = get_module(engine)
        recv_buf = ncp.zeros(send_buf.size, dtype=send_buf.dtype)
        base_comm.Allreduce(send_buf, recv_buf, op)
        return recv_buf
    else:
        # CuPy with non-CUDA-aware MPI
        if recv_buf is None:
            return base_comm.allreduce(send_buf, op)
        # For MIN and MAX which require recv_buf
        base_comm.Allreduce(send_buf, recv_buf, op)
        return recv_buf


def mpi_bcast(base_comm: MPI.Comm,
              rank, local_array, index, value,
              engine: Optional[str] = "numpy") -> np.ndarray:
    if deps.cuda_aware_mpi_enabled or engine == "numpy":
        if rank == 0:
            local_array[index] = value
        base_comm.Bcast(local_array[index])
    else:
        # CuPy with non-CUDA-aware MPI
        local_array[index] = base_comm.bcast(value)


def mpi_send(base_comm: MPI.Comm,
             send_buf, dest, count, tag=0,
             engine: Optional[str] = "numpy",
             ) -> None:
    """MPI_Send/send

    Dispatch send routine based on type of input and availability of
    CUDA-Aware MPI

    Parameters
    ----------
    base_comm : :obj:`MPI.Comm`
        Base MPI Communicator.
    send_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        The array containing data to send.
    dest: :obj:`int`
        The rank of the destination CPU/GPU device.
    count : :obj:`int`
        Number of elements to send from `send_buf`.
    tag : :obj:`int`
        Tag of the message to be sent.
    engine : :obj:`str`, optional
        Engine used to store array (``numpy`` or ``cupy``)
    """
    if deps.cuda_aware_mpi_enabled or engine == "numpy":
        # Determine MPI type based on array dtype
        mpi_type = MPI._typedict[send_buf.dtype.char]
        if count is None:
            count = send_buf.size
        base_comm.Send([send_buf, count, mpi_type], dest=dest, tag=tag)
    else:
        # Uses CuPy without CUDA-aware MPI
        base_comm.send(send_buf, dest, tag)


def mpi_recv(base_comm: MPI.Comm,
             recv_buf=None, source=0, count=None, tag=0,
             engine: Optional[str] = "numpy") -> np.ndarray:
    """ MPI_Recv/recv
    Dispatch receive routine based on type of input and availability of
    CUDA-Aware MPI

    Parameters
    ----------
    base_comm : :obj:`MPI.Comm`
        Base MPI Communicator.
    recv_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`, optional
        The buffered array to receive data.
    source : :obj:`int`
        The rank of the sending CPU/GPU device.
    count : :obj:`int`
        Number of elements to receive.
    tag : :obj:`int`
        Tag of the message to be sent.
    engine : :obj:`str`, optional
        Engine used to store array (``numpy`` or ``cupy``)
    """
    if deps.cuda_aware_mpi_enabled or engine == "numpy":
        ncp = get_module(engine)
        if recv_buf is None:
            if count is None:
                raise ValueError("Must provide either recv_buf or count for MPI receive")
            # Default to int32 works currently because add_ghost_cells() is called
            # with recv_buf and is not affected by this branch. The int32 is for when
            # dimension or shape-related integers are send/recv
            recv_buf = ncp.zeros(count, dtype=ncp.int32)
        mpi_type = MPI._typedict[recv_buf.dtype.char]
        base_comm.Recv([recv_buf, recv_buf.size, mpi_type], source=source, tag=tag)
    else:
        # Uses CuPy without CUDA-aware MPI
        recv_buf = base_comm.recv(source=source, tag=tag)
    return recv_buf
