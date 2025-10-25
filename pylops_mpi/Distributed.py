from typing import Any, NewType, Optional, Union

from mpi4py import MPI
from pylops.utils import NDArray
from pylops.utils import deps as pylops_deps  # avoid namespace crashes with pylops_mpi.utils
from pylops_mpi.utils._mpi import mpi_allreduce, mpi_allgather, mpi_bcast, mpi_send, mpi_recv, _prepare_allgather_inputs, _unroll_allgather_recv
from pylops_mpi.utils import deps

cupy_message = pylops_deps.cupy_import("the DistributedArray module")
nccl_message = deps.nccl_import("the DistributedArray module")

if nccl_message is None and cupy_message is None:
    from pylops_mpi.utils._nccl import (
        nccl_allgather, nccl_allreduce, nccl_bcast, nccl_send, nccl_recv
    )
    from cupy.cuda.nccl import NcclCommunicator
else:
    NcclCommunicator = Any

NcclCommunicatorType = NewType("NcclCommunicator", NcclCommunicator)


class DistributedMixIn:
    r"""Distributed Mixin class

    This class implements all methods associated with communication primitives
    from MPI and NCCL. It is mostly charged with identifying which commuicator
    to use and whether the buffered or object MPI primitives should be used
    (the former in the case of NumPy arrays or CuPy arrays when a CUDA-Aware
    MPI installation is available, the latter with CuPy arrays when a CUDA-Aware
    MPI installation is not available).

    """
    def _allreduce(self,
                   base_comm: MPI.Comm,
                   base_comm_nccl: NcclCommunicatorType,
                   send_buf: NDArray,
                   recv_buf: Optional[NDArray] = None,
                   op: MPI.Op = MPI.SUM,
                   engine: str = "numpy",
                   ) -> NDArray:
        """Allreduce operation

        Parameters
        ----------
        base_comm : :obj:`MPI.Comm`
            Base MPI Communicator.
        base_comm_nccl : :obj:`cupy.cuda.nccl.NcclCommunicator`
            NCCL Communicator.
        send_buf: :obj: `numpy.ndarray` or `cupy.ndarray`
            A buffer containing the data to be sent by this rank.
        recv_buf : :obj: `numpy.ndarray` or `cupy.ndarray`, optional
            The buffer to store the result of the reduction. If None,
            a new buffer will be allocated with the appropriate shape.
        op : :obj: `MPI.Op`, optional
            MPI operation to perform.
        engine : :obj:`str`, optional
            Engine used to store array (``numpy`` or ``cupy``)

        Returns
        -------
        recv_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
            A buffer containing the result of the reduction, broadcasted
            to all GPUs.

        """
        if deps.nccl_enabled and base_comm_nccl is not None:
            return nccl_allreduce(base_comm_nccl, send_buf, recv_buf, op)
        else:
            return mpi_allreduce(base_comm, send_buf,
                                 recv_buf, engine, op)

    def _allreduce_subcomm(self,
                           sub_comm: MPI.Comm,
                           base_comm_nccl: NcclCommunicatorType,
                           send_buf: NDArray,
                           recv_buf: Optional[NDArray] = None,
                           op: MPI.Op = MPI.SUM,
                           engine: str = "numpy",
                           ) -> NDArray:
        """Allreduce operation with subcommunicator

        Parameters
        ----------
        sub_comm : :obj:`MPI.Comm`
            MPI Subcommunicator.
        base_comm_nccl : :obj:`cupy.cuda.nccl.NcclCommunicator`
            NCCL Communicator.
        send_buf: :obj: `numpy.ndarray` or `cupy.ndarray`
            A buffer containing the data to be sent by this rank.
        recv_buf : :obj: `numpy.ndarray` or `cupy.ndarray`, optional
            The buffer to store the result of the reduction. If None,
            a new buffer will be allocated with the appropriate shape.
        op : :obj: `MPI.Op`, optional
            MPI operation to perform.
        engine : :obj:`str`, optional
            Engine used to store array (``numpy`` or ``cupy``)

        Returns
        -------
        recv_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
            A buffer containing the result of the reduction, broadcasted
            to all ranks.

        """
        if deps.nccl_enabled and base_comm_nccl is not None:
            return nccl_allreduce(sub_comm, send_buf, recv_buf, op)
        else:
            return mpi_allreduce(sub_comm, send_buf,
                                 recv_buf, engine, op)

    def _allgather(self,
                   base_comm: MPI.Comm,
                   base_comm_nccl: NcclCommunicatorType,
                   send_buf: NDArray,
                   recv_buf: Optional[NDArray] = None,
                   engine: str = "numpy",
                   ) -> NDArray:
        """Allgather operation

        Parameters
        ----------
        base_comm : :obj:`MPI.Comm`
            Base MPI Communicator.
        base_comm_nccl : :obj:`cupy.cuda.nccl.NcclCommunicator`
            NCCL Communicator.
        send_buf: :obj: `numpy.ndarray` or `cupy.ndarray`
            A buffer containing the data to be sent by this rank.
        recv_buf : :obj: `numpy.ndarray` or `cupy.ndarray`, optional
            The buffer to store the result of the gathering. If None,
            a new buffer will be allocated with the appropriate shape.
        engine : :obj:`str`, optional
            Engine used to store array (``numpy`` or ``cupy``)

        Returns
        -------
        recv_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
            A buffer containing the gathered data from all ranks.

        """
        if deps.nccl_enabled and base_comm_nccl is not None:
            if isinstance(send_buf, (tuple, list, int)):
                return nccl_allgather(base_comm_nccl, send_buf, recv_buf)
            else:
                send_shapes = base_comm.allgather(send_buf.shape)
                (padded_send, padded_recv) = _prepare_allgather_inputs(send_buf, send_shapes, engine="cupy")
                raw_recv = nccl_allgather(base_comm_nccl, padded_send, recv_buf if recv_buf else padded_recv)
                return _unroll_allgather_recv(raw_recv, padded_send.shape, send_shapes)
        else:
            if isinstance(send_buf, (tuple, list, int)):
                return base_comm.allgather(send_buf)
            return mpi_allgather(base_comm, send_buf, recv_buf, engine)

    def _allgather_subcomm(self,
                           sub_comm: MPI.Comm,
                           base_comm_nccl: NcclCommunicatorType,
                           send_buf: NDArray,
                           recv_buf: Optional[NDArray] = None,
                           engine: str = "numpy",
                           ) -> NDArray:
        """Allgather operation with subcommunicator

        Parameters
        ----------
        sub_comm : :obj:`MPI.Comm`
            MPI Subcommunicator.
        base_comm_nccl : :obj:`cupy.cuda.nccl.NcclCommunicator`
            NCCL Communicator.
        send_buf: :obj: `numpy.ndarray` or `cupy.ndarray`
            A buffer containing the data to be sent by this rank.
        recv_buf : :obj: `numpy.ndarray` or `cupy.ndarray`, optional
            The buffer to store the result of the gathering. If None,
            a new buffer will be allocated with the appropriate shape.
        engine : :obj:`str`, optional
            Engine used to store array (``numpy`` or ``cupy``)

        Returns
        -------
        recv_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
            A buffer containing the gathered data from all ranks.

        """
        if deps.nccl_enabled and base_comm_nccl is not None:
            if isinstance(send_buf, (tuple, list, int)):
                return nccl_allgather(sub_comm, send_buf, recv_buf)
            else:
                send_shapes = sub_comm._allgather_subcomm(send_buf.shape)
                (padded_send, padded_recv) = _prepare_allgather_inputs(send_buf, send_shapes, engine="cupy")
                raw_recv = nccl_allgather(sub_comm, padded_send, recv_buf if recv_buf else padded_recv)
                return _unroll_allgather_recv(raw_recv, padded_send.shape, send_shapes)
        else:
            return mpi_allgather(sub_comm, send_buf, recv_buf, engine)

    def _bcast(self,
               base_comm: MPI.Comm,
               base_comm_nccl: NcclCommunicatorType,
               rank : int,
               local_array: NDArray,
               index: int,
               value: Union[int, NDArray],
               engine: str = "numpy",
               ) -> None:
        """BCast operation

        Parameters
        ----------
        base_comm : :obj:`MPI.Comm`
            Base MPI Communicator.
        base_comm_nccl : :obj:`cupy.cuda.nccl.NcclCommunicator`
            NCCL Communicator.
        rank : :obj:`int`
            Rank.
        local_array : :obj:`numpy.ndarray`
            Localy array to be broadcasted.
        index : :obj:`int` or :obj:`slice`
            Represents the index positions where a value needs to be assigned.
        value : :obj:`int` or :obj:`numpy.ndarray`
            Represents the value that will be assigned to the local array at
            the specified index positions.
        engine : :obj:`str`, optional
            Engine used to store array (``numpy`` or ``cupy``)

        """
        if deps.nccl_enabled and base_comm_nccl is not None:
            nccl_bcast(base_comm_nccl, local_array, index, value)
        else:
            mpi_bcast(base_comm, rank, local_array, index, value,
                      engine=engine)

    def _send(self,
              base_comm: MPI.Comm,
              base_comm_nccl: NcclCommunicatorType,
              send_buf: NDArray,
              dest: int,
              count: Optional[int] = None,
              tag: int = 0,
              engine: str = "numpy",
              ) -> None:
        """Send operation

        Parameters
        ----------
        base_comm : :obj:`MPI.Comm`
            Base MPI Communicator.
        base_comm_nccl : :obj:`cupy.cuda.nccl.NcclCommunicator`
            NCCL Communicator.
        send_buf : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
            The array containing data to send.
        dest: :obj:`int`
            The rank of the destination.
        count : :obj:`int`
            Number of elements to send from `send_buf`.
        tag : :obj:`int`
            Tag of the message to be sent.
        engine : :obj:`str`, optional
            Engine used to store array (``numpy`` or ``cupy``)

        """
        if deps.nccl_enabled and base_comm_nccl is not None:
            if count is None:
                count = send_buf.size
            nccl_send(base_comm_nccl, send_buf, dest, count)
        else:
            mpi_send(base_comm,
                     send_buf, dest, count, tag=tag,
                     engine=engine)

    def _recv(self,
              base_comm: MPI.Comm,
              base_comm_nccl: NcclCommunicatorType,
              recv_buf=None, source=0, count=None, tag=0,
              engine: str = "numpy",
              ) -> NDArray:
        """Receive operation

        Parameters
        ----------
        base_comm : :obj:`MPI.Comm`
            Base MPI Communicator.
        base_comm_nccl : :obj:`cupy.cuda.nccl.NcclCommunicator`
            NCCL Communicator.
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
        if deps.nccl_enabled and base_comm_nccl is not None:
            if recv_buf is None:
                raise ValueError("recv_buf must be supplied when using NCCL")
            if count is None:
                count = recv_buf.size
            nccl_recv(base_comm_nccl, recv_buf, source, count)
            return recv_buf
        else:
            return mpi_recv(base_comm,
                            recv_buf, source, count, tag=tag,
                            engine=engine)
