import numpy as np
from scipy.sparse.linalg._interface import _get_dtype
from mpi4py import MPI
from typing import Optional, Sequence

from pylops import LinearOperator
from pylops.utils import DTypeLike
from pylops.utils.backend import get_module

from pylops_mpi import MPILinearOperator, MPIStackedLinearOperator
from pylops_mpi import DistributedArray, StackedDistributedArray
from pylops_mpi.utils.decorators import reshaped


class MPIBlockDiag(MPILinearOperator):
    r"""MPI Block-diagonal operator.

    Create a block-diagonal operator from a set of linear operators using MPI.
    Each rank must initialize this operator by providing one or more linear operators
    which will be computed within such rank.

    Both model and data vectors must be of :class:`pylops_mpi.DistributedArray` type and partitioned between ranks
    according to the shapes of the different linear operators.

    Parameters
    ----------
    ops : :obj:`list`
        One or more :class:`pylops.LinearOperator` to be stacked.
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        Base MPI Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape

    Notes
    -----
    An MPI Block Diagonal operator is composed of N linear operators, represented by **L**.
    Each rank has one or more :class:`pylops.LinearOperator`, which we represent here compactly
    as :math:`\mathbf{L}_i` for rank :math:`i`.

    Each operator performs forward mode operations using its corresponding model vector, denoted as **m**.
    This vector is effectively a :class:`pylops_mpi.DistributedArray` partitioned at each rank in such a way that
    its local shapes agree with those of the corresponding linear operators.
    The forward mode of each operator is then collected from all ranks as a DistributedArray, referred to as **d**.

    .. math::
        \begin{bmatrix}
        \mathbf{d}_1 \\
        \mathbf{d}_2 \\
        \vdots \\
        \mathbf{d}_n
        \end{bmatrix} =
        \begin{bmatrix}
        \mathbf{L}_1 & \mathbf{0} & \ldots & \mathbf{0} \\
        \mathbf{0} & \mathbf{L}_2 & \ldots & \mathbf{0} \\
        \vdots & \vdots & \ddots & \vdots \\
        \mathbf{0} & \mathbf{0} & \ldots & \mathbf{L}_n
        \end{bmatrix}
        \begin{bmatrix}
        \mathbf{m}_1 \\
        \mathbf{m}_2 \\
        \vdots \\
        \mathbf{m}_n
        \end{bmatrix}

    Likewise, for the adjoint mode, each operator executes operations in the adjoint mode,
    the adjoint mode of each operator is then collected from all ranks as a DistributedArray
    referred as **d**.

    .. math::
        \begin{bmatrix}
        \mathbf{d}_1 \\
        \mathbf{d}_2 \\
        \vdots \\
        \mathbf{d}_n
        \end{bmatrix} =
        \begin{bmatrix}
        \mathbf{L}_1^H & \mathbf{0} & \ldots & \mathbf{0} \\
        \mathbf{0} & \mathbf{L}_2^H & \ldots & \mathbf{0} \\
        \vdots & \vdots & \ddots & \vdots \\
        \mathbf{0} & \mathbf{0} & \ldots & \mathbf{L}_n^H
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
        mops = np.zeros(len(self.ops), dtype=np.int64)
        nops = np.zeros(len(self.ops), dtype=np.int64)
        for iop, oper in enumerate(self.ops):
            nops[iop] = oper.shape[0]
            mops[iop] = oper.shape[1]
        self.mops = mops.sum()
        self.local_shapes_m = base_comm.allgather((self.mops, ))
        self.nops = nops.sum()
        self.local_shapes_n = base_comm.allgather((self.nops, ))
        self.nnops = np.insert(np.cumsum(nops), 0, 0)
        self.mmops = np.insert(np.cumsum(mops), 0, 0)
        shape = (base_comm.allreduce(self.nops), base_comm.allreduce(self.mops))
        dtype = _get_dtype(ops) if dtype is None else np.dtype(dtype)
        super().__init__(shape=shape, dtype=dtype, base_comm=base_comm)

    @reshaped(forward=True, stacking=True)
    def _matvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        y = DistributedArray(global_shape=self.shape[0], local_shapes=self.local_shapes_n, 
                             engine=x.engine, dtype=self.dtype)
        y1 = []
        for iop, oper in enumerate(self.ops):
            y1.append(oper.matvec(x.local_array[self.mmops[iop]:
                                                self.mmops[iop + 1]]))
        y[:] = ncp.concatenate(ncp.asarray(y1))
        return y

    @reshaped(forward=False, stacking=True)
    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        y = DistributedArray(global_shape=self.shape[1], local_shapes=self.local_shapes_m,
                             engine=x.engine, dtype=self.dtype)
        y1 = []
        for iop, oper in enumerate(self.ops):
            y1.append(oper.rmatvec(x.local_array[self.nnops[iop]:
                                                 self.nnops[iop + 1]]))
        y[:] = ncp.concatenate(ncp.asarray(y1))
        return y


class MPIStackedBlockDiag(MPIStackedLinearOperator):
    r"""MPI Stacked BlockDiag Operator

    Create a diagonal stack of :class:`pylops_mpi.MPILinearOperator` operators.

    Parameters
    ----------
    ops : :obj:`list`
        One or more :class:`pylops_mpi.MPILinearOperator` to be stacked.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape

    Notes
    -----
    A MPIStackedBlockDiag is composed of N :class:pylops_mpi.MPILinearOperator instances stacked along the diagonal.
    These MPI operators will be applied sequentially, with distributed computations
    performed within each operator.

    """

    def __init__(self, ops: Sequence[MPILinearOperator], base_comm: MPI.Comm = MPI.COMM_WORLD,
                 dtype: Optional[DTypeLike] = None):
        self.ops = ops
        dtype = _get_dtype(self.ops) if dtype is None else np.dtype(dtype)
        shape = (int(np.sum(op.shape[0] for op in ops)),
                 int(np.sum(op.shape[1] for op in ops)))
        super().__init__(shape=shape, dtype=dtype, base_comm=base_comm)

    def _matvec(self, x: StackedDistributedArray) -> StackedDistributedArray:
        y1 = []
        for xx, oper in zip(x, self.ops):
            y1.append(oper.matvec(xx))
        return StackedDistributedArray(y1)

    def _rmatvec(self, x: StackedDistributedArray) -> StackedDistributedArray:
        y1 = []
        for xx, oper in zip(x, self.ops):
            y1.append(oper.rmatvec(xx))
        return StackedDistributedArray(y1)
