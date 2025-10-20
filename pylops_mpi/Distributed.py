from mpi4py import MPI
from pylops.utils import deps as pylops_deps  # avoid namespace crashes with pylops_mpi.utils
from pylops_mpi.utils._mpi import mpi_allreduce, mpi_allgather, mpi_bcast, mpi_send, mpi_recv, _prepare_allgather_inputs, _unroll_allgather_recv
from pylops_mpi.utils import deps

cupy_message = pylops_deps.cupy_import("the DistributedArray module")
nccl_message = deps.nccl_import("the DistributedArray module")

if nccl_message is None and cupy_message is None:
    from pylops_mpi.utils._nccl import (
        nccl_allgather, nccl_allreduce, nccl_bcast, nccl_send, nccl_recv
    )


class DistributedMixIn:
    r"""Distributed Mixin class

    This class implements all methods associated with communication primitives
    from MPI and NCCL. It is mostly charged with identifying which commuicator
    to use and whether the buffered or object MPI primitives should be used
    (the former in the case of NumPy arrays or CuPy arrays when a CUDA-Aware
    MPI installation is available, the latter with CuPy arrays when a CUDA-Aware
    MPI installation is not available).

    """
    def _allreduce(self, base_comm, base_comm_nccl,
                   send_buf, recv_buf=None,
                   op: MPI.Op = MPI.SUM,
                   engine="numpy"):
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

    def _allreduce_subcomm(self, sub_comm, base_comm_nccl,
                           send_buf, recv_buf=None,
                           op: MPI.Op = MPI.SUM,
                           engine="numpy"):
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

    def _allgather(self, base_comm, base_comm_nccl,
                   send_buf, recv_buf=None,
                   engine="numpy"):
        """Allgather operation

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

    def _allgather_subcomm(self, send_buf, recv_buf=None):
        """Allgather operation with subcommunicator
        """
        if deps.nccl_enabled and getattr(self, "base_comm_nccl"):
            if isinstance(send_buf, (tuple, list, int)):
                return nccl_allgather(self.sub_comm, send_buf, recv_buf)
            else:
                send_shapes = self._allgather_subcomm(send_buf.shape)
                (padded_send, padded_recv) = _prepare_allgather_inputs(send_buf, send_shapes, engine="cupy")
                raw_recv = nccl_allgather(self.sub_comm, padded_send, recv_buf if recv_buf else padded_recv)
                return _unroll_allgather_recv(raw_recv, padded_send.shape, send_shapes)
        else:
            return mpi_allgather(self.sub_comm, send_buf, recv_buf, self.engine)

    def _bcast(self, local_array, index, value):
        """BCast operation
        """
        if deps.nccl_enabled and getattr(self, "base_comm_nccl"):
            nccl_bcast(self.base_comm_nccl, local_array, index, value)
        else:
            # self.local_array[index] = self.base_comm.bcast(value)
            mpi_bcast(self.base_comm, self.rank, self.local_array, index, value,
                      engine=self.engine)

    def _send(self, send_buf, dest, count=None, tag=0):
        """Send operation
        """
        if deps.nccl_enabled and self.base_comm_nccl:
            if count is None:
                count = send_buf.size
            nccl_send(self.base_comm_nccl, send_buf, dest, count)
        else:
            mpi_send(self.base_comm,
                     send_buf, dest, count, tag=tag,
                     engine=self.engine)

    def _recv(self, recv_buf=None, source=0, count=None, tag=0):
        """Receive operation
        """
        if deps.nccl_enabled and self.base_comm_nccl:
            if recv_buf is None:
                raise ValueError("recv_buf must be supplied when using NCCL")
            if count is None:
                count = recv_buf.size
            nccl_recv(self.base_comm_nccl, recv_buf, source, count)
            return recv_buf
        else:
            return mpi_recv(self.base_comm,
                            recv_buf, source, count, tag=tag,
                            engine=self.engine)
