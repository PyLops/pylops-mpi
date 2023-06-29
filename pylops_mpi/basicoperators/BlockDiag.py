import numpy as np
from scipy.sparse.linalg._interface import _get_dtype
from mpi4py import MPI
from typing import Optional, Sequence

from pylops.utils import DTypeLike
from pylops import LinearOperator, MatrixMult

from pylops_mpi.DistributedArray import local_split, Partition


class MPIBlockDiag(LinearOperator):
    r"""MPI Block-diagonal operator.

        Create a block-diagonal operator from N linear operators using MPI.

        Parameters
        ----------
        ops : :obj:`list`
            Linear operators to be stacked. Alternatively,
            :obj:`numpy.ndarray` or :obj:`scipy.sparse` matrices can be passed
            in place of one or more operators.
        base_comm : :obj:`mpi4py.MPI.Comm`, optional
            Base MPI Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
        dtype : :obj:`str`, optional
            Type of elements in input array.

        Attributes
        ----------
        shape : :obj:`tuple`
            Operator shape
        explicit : :obj:`bool`
            Operator contains a matrix that can be solved explicitly (``True``) or
            not (``False``)

        Notes
        -----
        Refer to :obj:pylops.BlockDiag for more details.
    """

    def __init__(self, ops: Sequence[LinearOperator],
                 base_comm: MPI.Comm = MPI.COMM_WORLD,
                 dtype: Optional[DTypeLike] = None):
        # Required for MPI
        self.base_comm = base_comm
        self.rank = base_comm.Get_rank()
        self.size = base_comm.Get_size()
        # Assign ops to different ranks
        self.ops = self._assign_ops(ops)
        mops = np.zeros(len(self.ops), dtype=np.int64)
        nops = np.zeros(len(self.ops), dtype=np.int64)
        for iop, oper in enumerate(self.ops):
            if not isinstance(oper, LinearOperator):
                self.ops[iop] = MatrixMult(oper, dtype=oper.dtype)
            nops[iop] = oper.shape[0]
            mops[iop] = oper.shape[1]
        self.mops = mops.sum()
        self.nops = nops.sum()
        self.nnops = np.insert(np.cumsum(nops), 0, 0)
        self.mmops = np.insert(np.cumsum(mops), 0, 0)
        # Shape of the operator is equal to the sum of nops and mops at all ranks
        self.shape = (base_comm.allreduce(self.nops), base_comm.allreduce(self.mops))
        if dtype:
            self.dtype = _get_dtype(dtype)
        else:
            self.dtype = np.dtype(dtype)
        clinear = all([getattr(oper, "clinear", True) for oper in self.ops])
        super().__init__(shape=self.shape, dtype=self.dtype, clinear=clinear)

    def _assign_ops(self, ops):
        r"""Assign operators to different ranks

           Parameters
           ----------
           ops : :obj:`list`
               Linear operators to be stacked.

           Returns
           -------
           ops : :obj:`list`
               Linear Operators allocated at each rank.
        """
        local_shapes = local_split((len(ops),), self.base_comm, Partition.SCATTER, 0)
        x = np.insert(np.cumsum(self.base_comm.allgather(local_shapes)), 0, 0)
        return ops[x[self.rank]:x[self.rank + 1]]

    def _matvec(self, x):
        # Calculate x_local using mops for different ranks
        mops_sum = np.insert(np.cumsum(self.base_comm.allgather(self.mops)), 0, 0)
        x_local = x[mops_sum[self.rank]: mops_sum[self.rank + 1]]
        y = []
        for iop, oper in enumerate(self.ops):
            y.append(oper.matvec(x_local[self.mmops[iop]:
                                         self.mmops[iop + 1]]))
        if y:
            y = np.concatenate(y)
        return np.hstack(self.base_comm.allgather(y))

    def _rmatvec(self, x):
        # Calculate x_local using nops for different ranks
        nops_sum = np.insert(np.cumsum(self.base_comm.allgather(self.nops)), 0, 0)
        x_local = x[nops_sum[self.rank]: nops_sum[self.rank + 1]]
        y = []
        for iop, oper in enumerate(self.ops):
            y.append(oper.rmatvec(x_local[self.nnops[iop]:
                                          self.nnops[iop + 1]]))
        if y:
            y = np.concatenate(y)
        return np.hstack(self.base_comm.allgather(y))
