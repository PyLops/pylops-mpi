from typing import Any, NewType

from mpi4py import MPI
from pylops.utils import deps as pylops_deps  # avoid namespace crashes with pylops_mpi.utils
from pylops_mpi.utils._mpi import mpi_allreduce, mpi_send
from pylops_mpi.utils import deps

cupy_message = pylops_deps.cupy_import("the DistributedArray module")
nccl_message = deps.nccl_import("the DistributedArray module")

if nccl_message is None and cupy_message is None:
    from pylops_mpi.utils._nccl import (
        nccl_allgather, nccl_allreduce, 
        nccl_asarray, nccl_bcast, nccl_split, nccl_send, nccl_recv,
        _prepare_nccl_allgather_inputs, _unroll_nccl_allgather_recv
    )


class DistributedMixIn:
    r"""Distributed Mixin class

    This class implements all methods associated with communication primitives
    from MPI and NCCL. It is mostly charged to identifying which commuicator
    to use and whether the buffered or object MPI primitives should be used
    (the former in the case of NumPy arrays or CuPy arrays when a CUDA-Aware
    MPI installation is available, the latter with CuPy arrays when a CUDA-Aware
    MPI installation is not available).
    """
    def _allreduce(self, send_buf, recv_buf=None, op: MPI.Op = MPI.SUM):
        """Allreduce operation
        """
        if deps.nccl_enabled and getattr(self, "base_comm_nccl"):
            return nccl_allreduce(self.base_comm_nccl, send_buf, recv_buf, op)
        else:
            return mpi_allreduce(self.base_comm, send_buf, 
                                 recv_buf, self.engine, op)

    def _allreduce_subcomm(self, send_buf, recv_buf=None, op: MPI.Op = MPI.SUM):
        """Allreduce operation with subcommunicator
        """
        if deps.nccl_enabled and getattr(self, "base_comm_nccl"):
            return nccl_allreduce(self.sub_comm, send_buf, recv_buf, op)
        else:
            return mpi_allreduce(self.sub_comm, send_buf, 
                                 recv_buf, self.engine, op)

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
