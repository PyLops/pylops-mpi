import numpy as np
import scipy as sp
from mpi4py import MPI
from typing import Optional

from pylops.utils import DTypeLike
from pylops_mpi import MPILinearOperator, DistributedArray
from pylops_mpi.DistributedArray import local_split, Partition

sp_version = sp.__version__.split(".")
if int(sp_version[0]) <= 1 and int(sp_version[1]) < 8:
    from scipy.sparse.linalg.interface import _get_dtype
else:
    from scipy.sparse.linalg._interface import _get_dtype


class MPIBlockDiag(MPILinearOperator):

    def __init__(self, ops, base_comm: MPI.Comm = MPI.COMM_WORLD, dtype: Optional[DTypeLike] = None):
        self.ops = self._assign_ops(ops, base_comm)
        self.kind = "mix"
        mops = np.zeros(len(self.ops), dtype=np.int64)
        nops = np.zeros(len(self.ops), dtype=np.int64)
        for iop, oper in enumerate(self.ops):
            nops[iop] = oper.shape[0]
            mops[iop] = oper.shape[1]
        self.mops = base_comm.allgather((mops.sum(), ))
        self.nops = base_comm.allgather((nops.sum(), ))
        self.nnops = np.insert(np.cumsum(nops), 0, 0)
        self.mmops = np.insert(np.cumsum(mops), 0, 0)
        self.shape = (np.sum(self.nops), np.sum(self.mops))
        if dtype:
            self.dtype = _get_dtype(dtype)
        else:
            self.dtype = np.dtype(dtype)
        super().__init__(shape=self.shape, dtype=self.dtype, base_comm=base_comm, kind=self.kind)

    @staticmethod
    def _assign_ops(ops, base_comm):
        local_shapes = local_split((len(ops),), base_comm, Partition.SCATTER, 0)
        x = np.insert(np.cumsum(base_comm.allgather(local_shapes)), 0, 0)
        return ops[x[base_comm.Get_rank()]:x[base_comm.Get_rank() + 1]]

    def _matvec(self, x):
        y = DistributedArray(global_shape=self.shape[0], local_shapes=self.nops)
        if not isinstance(x, DistributedArray):
            x = DistributedArray.to_dist(x=x, local_shapes=self.mops)
        y1 = []
        for iop, oper in enumerate(self.ops):
            y1.append(oper.matvec(x.local_array[self.mmops[iop]:
                                                self.mmops[iop + 1]]))
        if y1:
            y[:] = np.concatenate(y1)
        return y

    def _rmatvec(self, x):
        y = DistributedArray(global_shape=self.shape[1], local_shapes=self.mops)
        if not isinstance(x, DistributedArray):
            x = DistributedArray.to_dist(x=x, local_shapes=self.nops)
        y1 = []
        for iop, oper in enumerate(self.ops):
            y1.append(oper.rmatvec(x.local_array[self.nnops[iop]: self.nnops[iop + 1]]))
        if y1:
            y[:] = np.concatenate(y1)
        return y
