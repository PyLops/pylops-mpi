__all__ = [
    "cupy_to_nccl_dtype",
    "mpi_op_to_nccl",
    "initialize_nccl_comm",
    "get_cupy"
]

from types import ModuleType

from enum import IntEnum
from mpi4py import MPI
from pylops.utils.deps import cupy_enabled
from pylops_mpi.utils import deps
import os

cupy_to_nccl_dtype = {}
if cupy_enabled:
    import cupy as cp
else:
    cp = None

if deps.nccl_enabled:
    import cupy.cuda.nccl as nccl

    cupy_to_nccl_dtype.update({
        "float32": nccl.NCCL_FLOAT32,
        "float64": nccl.NCCL_FLOAT64,
        "int32": nccl.NCCL_INT32,
        "int64": nccl.NCCL_INT64,
        "uint8": nccl.NCCL_UINT8,
        "int8": nccl.NCCL_INT8,
        "uint32": nccl.NCCL_UINT32,
        "uint64": nccl.NCCL_UINT64,
    })

    class NcclOp(IntEnum):
        SUM = nccl.NCCL_SUM
        PROD = nccl.NCCL_PROD
        MAX = nccl.NCCL_MAX
        MIN = nccl.NCCL_MIN

    def mpi_op_to_nccl(mpi_op) -> NcclOp:
        """ Map MPI reduction operation to NCCL equivalent
        Parameters
        ----------
        mpi_op (MPI.Op): An MPI reduction operation (e.g., MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN).

        Returns:
        -------
        NcclOp: The corresponding NCCL reduction operation.
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
        """ Initilize NCCL world communicator for every GPU device
        Each GPU must be managed by exactly one MPI process.
        i.e. the number of MPI process launched must be equal to
        number of GPUs in communications

        Returns:
        -------
        cupy.cuda.nccl.NcclCommunicator: The corresponding NCCL communicator
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        local_rank = int(
            os.environ.get("SLURM_LOCALID")
            or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
            or rank % cp.cuda.runtime.getDeviceCount()
        )
        device_id = local_rank
        cp.cuda.Device(device_id).use()

        if rank == 0:
            nccl_id = nccl.get_unique_id()
        else:
            nccl_id = None
        nccl_id = comm.bcast(nccl_id, root=0)

        nccl_comm = nccl.NcclCommunicator(size, nccl_id, rank)
        return nccl_comm


def get_cupy() -> ModuleType:
    return cp
