import numpy as np
from scipy.sparse.linalg._interface import _get_dtype
from mpi4py import MPI
from typing import Sequence, Optional

from pylops import LinearOperator
from pylops.utils import DTypeLike

from pylops_mpi import MPILinearOperator, DistributedArray, Partition
from pylops_mpi.utils.decorators import reshaped


class MPIVStack(MPILinearOperator):
    r"""MPI VStack Operator

    Create a vertical stack of a set of linear operators using MPI. Each rank must
    initialize this operator by providing one or more linear operators which will
    be computed within each rank. Both model and data vectors are of
    :class:`pylops_mpi.DistributedArray` type.

    Parameters
    ----------
    ops : :obj:`list`
        One or more :class:`pylops.LinearOperator` to be vertically stacked.
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
        If ``ops`` have different number of columns

    Notes
    -----
    An MPIVStack is composed of N linear operators stacked vertically, represented by **L**.
    Each rank has one or more :class:`pylops.LinearOperator`, which we represent here compactly
    as :math:`\mathbf{L}_i` for rank :math:`i`.

    Each operator performs forward mode operations using its corresponding model vector, denoted as **m**.
    This vector is effectively a :class:`pylops_mpi.DistributedArray` with :obj:`pylops_mpi.Partition.BROADCAST`
    i.e. broadcasted at all ranks in such a way that the global shape is equal to the local shape of the array
    and these local shapes agree with the corresponding linear operators.

    Afterwards, the forward mode of each operator is collected from each rank in a
    :class:`pylops_mpi.DistributedArray`, represented by **d**.

    .. math::
          \begin{bmatrix}
            \mathbf{d}_1 \\
            \mathbf{d}_2 \\
            \vdots \\
            \mathbf{d}_n
          \end{bmatrix} =
          \begin{bmatrix}
            \mathbf{L}_1 \\
            \mathbf{L}_2 \\
            \vdots \\
            \mathbf{L}_n
          \end{bmatrix}
          m

    Likewise for the adjoint mode, each operator performs adjoint operations using its corresponding
    model vector, denoted as **m** which is a :class:`pylops_mpi.DistributedArray` with
    :obj:`pylops_mpi.Partition.SCATTER` i.e. partitioned at each rank.

    Afterwards, a collective reduction operation is performed where adjoint values from linear operators
    are summed up, and the final sum is broadcasted to all processes in the communicator, represented by **d**.

    .. math::
          d =
          \begin{bmatrix}
            \mathbf{L}_1^H &
            \mathbf{L}_2^H &
            \ldots &
            \mathbf{L}_n^H
          \end{bmatrix}
          \begin{bmatrix}
            \mathbf{m}_1 \\
            \mathbf{m}_2 \\
            \vdots \\
            \mathbf{m}_n
          \end{bmatrix}

    """

    def __init__(self, ops: Sequence[LinearOperator],
                 base_comm: MPI.Comm = MPI.COMM_WORLD,
                 dtype: Optional[DTypeLike] = None):
        self.ops = ops
        nops = np.zeros(len(self.ops), dtype=np.int64)
        for iop, oper in enumerate(self.ops):
            nops[iop] = oper.shape[0]
        self.nops = nops.sum()
        self.local_shapes_n = base_comm.allgather((self.nops, ))
        mops = [oper.shape[1] for oper in self.ops]
        mops = np.unique(base_comm.allgather(mops))
        if len(set(mops)) > 1:
            raise ValueError("Operators have different number of columns")
        self.mops = int(mops[0])
        self.nnops = np.insert(np.cumsum(nops), 0, 0)
        shape = (base_comm.allreduce(self.nops), self.mops)
        dtype = _get_dtype(self.ops) if dtype is None else np.dtype(dtype)
        super().__init__(shape=shape, dtype=dtype, base_comm=base_comm)

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        if x.partition is not Partition.BROADCAST:
            raise ValueError(f"x should have partition={Partition.BROADCAST}, {x.partition} != {Partition.BROADCAST}")
        y = DistributedArray(global_shape=self.shape[0], local_shapes=self.local_shapes_n, dtype=self.dtype)
        y1 = []
        for iop, oper in enumerate(self.ops):
            y1.append(oper.matvec(x.local_array))
        y[:] = np.concatenate(y1)
        return y

    @reshaped(forward=False, stacking=True)
    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        y = DistributedArray(global_shape=self.shape[1], partition=Partition.BROADCAST, dtype=self.dtype)
        y1 = []
        for iop, oper in enumerate(self.ops):
            y1.append(oper.rmatvec(x.local_array[self.nnops[iop]: self.nnops[iop + 1]]))
        y1 = np.sum(y1, axis=0)
        y[:] = self.base_comm.allreduce(y1, op=MPI.SUM)
        return y
