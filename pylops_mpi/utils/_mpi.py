__all__ = [
    "mpi_allgather",
    "mpi_allreduce",
    "mpi_bcast",
    "mpi_send",
    "mpi_recv",
    "mpi_sendrecv"
]

from typing import Optional

from mpi4py import MPI
from pylops.utils import NDArray
from pylops.utils.backend import get_module
from pylops_mpi.utils import deps


def mpi_allgather(base_comm: MPI.Comm,
                  send_buf: NDArray,
                  recv_buf: Optional[NDArray] = None,
                  engine: str = "numpy",
                  ) -> NDArray:
    """MPI_Allallgather/allallgather

    Dispatch allgather routine based on type of input and availability of
    CUDA-Aware MPI

    Parameters
    ----------
    base_comm : :obj:`MPI.Comm`
        Base MPI Communicator.
    send_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        The data buffer from the local rank to be gathered.
    recv_buf : :obj:`cupy.ndarray`, optional
        The buffer to store the result of the gathering. If None,
        a new buffer will be allocated with the appropriate shape.
    engine : :obj:`str`, optional
        Engine used to store array (``numpy`` or ``cupy``)

    Returns
    -------
    recv_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        A buffer containing the gathered data from all ranks.

    """
    if deps.cuda_aware_mpi_enabled or engine == "numpy":
        ncp = get_module(engine)
        send_shapes = base_comm.allgather(send_buf.shape)
        recvcounts = base_comm.allgather(send_buf.size)
        recv_buf = recv_buf if recv_buf else ncp.zeros(sum(recvcounts), dtype=send_buf.dtype)
        if len(set(send_shapes)) == 1:
            _mpi_calls(base_comm, "Allgather", send_buf.copy(), recv_buf, engine=engine)
            return [chunk.reshape(send_shapes[0]) for chunk in ncp.split(recv_buf, base_comm.size)]
        displs = [0]
        for i in range(1, len(recvcounts)):
            displs.append(displs[i - 1] + recvcounts[i - 1])
        _mpi_calls(base_comm, "Allgatherv", send_buf.copy(),
                   [recv_buf, recvcounts, displs, MPI._typedict[send_buf.dtype.char]], engine=engine)
        return [
            recv_buf[displs[i]:displs[i] + recvcounts[i]].reshape(send_shapes[i])
            for i in range(base_comm.size)
        ]
    else:
        # CuPy with non-CUDA-aware MPI
        if recv_buf is None:
            return _mpi_calls(base_comm, "allgather", send_buf)
        _mpi_calls(base_comm, "Allgather", send_buf, recv_buf)
        return recv_buf


def mpi_allreduce(base_comm: MPI.Comm,
                  send_buf: NDArray,
                  recv_buf: Optional[NDArray] = None,
                  engine: str = "numpy",
                  op: MPI.Op = MPI.SUM,
                  ) -> NDArray:
    """MPI_Allreduce/allreduce

    Dispatch allreduce routine based on type of input and availability of
    CUDA-Aware MPI

    Parameters
    ----------
    base_comm : :obj:`MPI.Comm`
        Base MPI Communicator.
    send_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        The data buffer from the local rank to be reduced.
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
        to all ranks.

    """
    if deps.cuda_aware_mpi_enabled or engine == "numpy":
        ncp = get_module(engine)
        recv_buf = ncp.zeros(send_buf.size, dtype=send_buf.dtype)
        _mpi_calls(base_comm, "Allreduce", send_buf, recv_buf, op, engine=engine)
        return recv_buf
    else:
        # CuPy with non-CUDA-aware MPI
        if recv_buf is None:
            return _mpi_calls(base_comm, "allreduce", send_buf, op, engine=engine)
        # For MIN and MAX which require recv_buf
        _mpi_calls(base_comm, "Allreduce", send_buf, recv_buf, op, engine=engine)
        return recv_buf


def mpi_bcast(base_comm: MPI.Comm,
              send_buf: NDArray,
              root: int = 0,
              engine: Optional[str] = "numpy",
              ) -> NDArray:
    """MPI_Bcast/bcast

    Dispatch bcast routine based on type of input and availability of
    CUDA-Aware MPI.

    Parameters
    ----------
    base_comm : :obj:`MPI.Comm`
        Base MPI Communicator.
    send_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        The data buffer to be broadcasted to the other ranks from the broadcasting root rank.
    root : :obj:`int`, optional
        The rank of the broadcasting process.
    engine : :obj:`str`, optional
        Engine used to store array (``numpy`` or ``cupy``)

    Returns
    -------
    send_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        The buffer containing the broadcasted data.

    """
    if deps.cuda_aware_mpi_enabled or engine == "numpy":
        _mpi_calls(base_comm, "Bcast", send_buf, engine=engine, root=root)
        return send_buf
    # CuPy with non-CUDA-aware MPI: use object broadcast
    value = send_buf if base_comm.Get_rank() == root else None
    return _mpi_calls(base_comm, "bcast", value, engine=engine, root=root)


def mpi_send(base_comm: MPI.Comm,
             send_buf: NDArray,
             dest: int,
             count: Optional[int] = None,
             tag: int = 0,
             engine: str = "numpy",
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
        _mpi_calls(base_comm, "Send", [send_buf, count, mpi_type], engine=engine, dest=dest, tag=tag)
    else:
        # Uses CuPy without CUDA-aware MPI
        _mpi_calls(base_comm, "send", send_buf, dest, tag, engine=engine)


def mpi_recv(base_comm: MPI.Comm,
             recv_buf: Optional[NDArray] = None,
             source: int = 0,
             count: Optional[int] = None,
             tag: int = 0,
             engine: Optional[str] = "numpy",
             ) -> NDArray:
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

    Returns
    -------
    recv_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        The buffer containing the received data.

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
        _mpi_calls(base_comm, "Recv", [recv_buf, recv_buf.size, mpi_type], engine=engine, source=source, tag=tag)
    else:
        # Uses CuPy without CUDA-aware MPI
        recv_buf = _mpi_calls(base_comm, "recv", engine=engine, source=source, tag=tag)
    return recv_buf


def mpi_sendrecv(base_comm: MPI.Comm, sendbuf: NDArray, recvbuf: NDArray, dest: int = 0, sendtag: int = 0,
                 source: int = 0, recvtag: int = 0, engine: Optional[str] = "numpy") -> NDArray:
    """MPI Send/Recv
    Dispatch send and receive in one combined call based on type of input and availability of
    CUDA-Aware MPI

    Parameters
    ----------
    base_comm : :obj:`MPI.Comm`
        Base MPI Communicator.
    sendbuf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        The array containing data to send.
    recvbuf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`, optional
        The buffered array to receive data.
    dest: :obj:`int`
        The rank of the destination CPU/GPU device.
    sendtag : :obj:`int`
        Tag of the message to be sent.
    source: :obj:`int`
        The rank of the sending CPU/GPU device.
    recvtag : :obj:`int`
        Tag of the message to be received.
    engine : :obj:`str`, optional
        Engine used to store array (``numpy`` or ``cupy``)

    Returns
    -------
    recv_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        The buffer containing the received data.

    """
    if deps.cuda_aware_mpi_enabled or engine == "numpy":
        _mpi_calls(base_comm, "Sendrecv", engine=engine, sendbuf=sendbuf, dest=dest, sendtag=sendtag,
                   recvbuf=recvbuf, source=source, recvtag=recvtag)
    else:
        recvbuf = _mpi_calls(base_comm, "sendrecv", engine=engine, sendobj=sendbuf, dest=dest, sendtag=sendtag,
                             source=source, recvtag=recvtag)
    return recvbuf


def _mpi_calls(comm: MPI.Comm, func: str, *args, engine: Optional[str] = "numpy", **kwargs):
    """MPI Calls
    Wrapper around MPI comm calls with optional GPU synchronization for CuPy arrays.

    Parameters
    ----------
    comm: :obj:`MPI.Comm`
        MPI Communicator
    func
        MPI Function to call.
    args
        Arguments to pass to the function.
    engine: :obj:`str`, optional
        Engine used to store array (``numpy`` or ``cupy``)
    kwargs
        Keyword arguments passed to the MPI call.

    Returns
    -------
        Result of the MPI call
    """
    if engine == "cupy" and deps.cuda_aware_mpi_enabled:
        ncp = get_module(engine)
        ncp.cuda.Device().synchronize()
    mpi_func = getattr(comm, func)
    return mpi_func(*args, **kwargs)
