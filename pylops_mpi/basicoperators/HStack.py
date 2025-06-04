import numpy as np
from typing import Sequence, Optional
from mpi4py import MPI
from pylops import LinearOperator
from pylops.utils import DTypeLike

from pylops_mpi import DistributedArray, MPILinearOperator
from pylops_mpi.DistributedArray import NcclCommunicatorType
from .VStack import MPIVStack


class MPIHStack(MPILinearOperator):
    r"""MPI HStack Operator

    Create a horizontal stack of a set of linear operators using MPI. Each rank must
    initialize this operator by providing one or more linear operators which will
    be computed within each rank. Both model and data vectors are of
    :class:`pylops_mpi.DistributedArray` type.

    Parameters
    ----------
    ops : :obj:`list`
        One or more :class:`pylops.LinearOperator` to be horizontally stacked.
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        Base MPI Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape

    Raises
    ------
    ValueError
        If ``ops`` have different number of rows

    Notes
    -----
    An MPIHStack is composed of N linear operators stacked horizontally, represented by **L**.
    Each rank has one or more :class:`pylops.LinearOperator`, which we represent here compactly
    as :math:`\mathbf{L}_i` for rank :math:`i`.

    Each operator performs forward mode operations using its corresponding model vector, denoted as **m**.
    This vector is effectively a :class:`pylops_mpi.DistributedArray`, with partition set to
    :obj:`pylops_mpi.Partition.SCATTER` i.e. unique portions assigned to each rank.

    Afterwards, a collective reduction operation is performed where matrix-vector product values from linear operators
    are summed up, and broadcasted to all ranks in the communicator, represented by **d**.

    .. math::
          d =
          \begin{bmatrix}
            \mathbf{L}_1 &
            \mathbf{L}_2 &
            \ldots &
            \mathbf{L}_n
          \end{bmatrix}
          \begin{bmatrix}
            \mathbf{m}_1 \\
            \mathbf{m}_2 \\
            \vdots \\
            \mathbf{m}_n
          \end{bmatrix}

    For the adjoint mode, each operator performs adjoint matrix-vector product using its corresponding
    model vector, denoted by **m** which is a :class:`pylops_mpi.DistributedArray` with partition set to
    :obj:`pylops_mpi.Partition.BROADCAST` i.e. array is broadcasted to all ranks.

    Afterwards, the result of the adjoint matrix-vector product is stored in a :obj:`pylops_mpi.DistributedArray`,
    which is represented by the variable **d**.

    .. math::
          \begin{bmatrix}
            \mathbf{d}_1 \\
            \mathbf{d}_2 \\
            \vdots \\
            \mathbf{d}_n
          \end{bmatrix} =
          \begin{bmatrix}
            \mathbf{L}_1^H \\
            \mathbf{L}_2^H \\
            \vdots \\
            \mathbf{L}_n^H
          \end{bmatrix}
          m

    """

    def __init__(self, ops: Sequence[LinearOperator],
                 base_comm: MPI.Comm = MPI.COMM_WORLD,
                 base_comm_nccl: NcclCommunicatorType = None,
                 dtype: Optional[DTypeLike] = None):
        self.ops = ops
        nops = [oper.shape[0] for oper in self.ops]
        nops = np.concatenate(base_comm.allgather(nops), axis=0)
        if len(set(nops)) > 1:
            raise ValueError("Operators have different number of rows")
        hops = [oper.H for oper in self.ops]
        self.HStack = MPIVStack(ops=hops, base_comm=base_comm, base_comm_nccl=base_comm_nccl, dtype=dtype).H
        super().__init__(shape=self.HStack.shape, dtype=self.HStack.dtype, base_comm=base_comm)

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        return self.HStack.matvec(x)

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        return self.HStack.rmatvec(x)
