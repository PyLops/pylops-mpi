from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI

from pylops import LinearOperator, MatrixMult
from pylops.utils import NDArray
from typing import Optional
import numpy as np
from scipy.sparse.linalg import LinearOperator as spLinearOperator

comm = MPI.COMM_WORLD


def _matvec_rmatvec_map(op, x: NDArray) -> NDArray:
    return op(x).squeeze()


class MPIVStack:

    def __init__(self, ops, max_workers: int = 1,
                 dtype: Optional[str] = None) -> None:
        self.ops = ops
        nops = np.zeros(len(self.ops), dtype=int)
        for iop, oper in enumerate(ops):
            if not isinstance(oper, (LinearOperator, spLinearOperator)):
                self.ops[iop] = MatrixMult(oper, dtype=oper.dtype)
            nops[iop] = self.ops[iop].shape[0]
        self.nops = int(nops.sum())
        mops = [oper.shape[1] for oper in self.ops]
        if len(set(mops)) > 1:
            raise ValueError("operators have different number of columns")
        self.mops = int(mops[0])
        self.nnops = np.insert(np.cumsum(nops), 0, 0)
        self.mpiexec = None
        self._max_workers = max_workers
        if self._max_workers > 1:
            self.mpiexec = MPIPoolExecutor(max_workers=max_workers)

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @max_workers.setter
    def max_workers(self, new_workers: int):
        if self._max_workers > 1:
            self.mpiexec.shutdown()
        if new_workers > 1:
            self.mpiexec = MPIPoolExecutor(max_workers=new_workers)
        self._max_workers = new_workers

    def _rmatvec_multiproc(self, x: NDArray) -> NDArray:
        y1 = None
        if comm.Get_rank() == 0:
            y1 = self.mpiexec.starmap(
                _matvec_rmatvec_map,
                [
                    (oper._rmatvec, x[self.nnops[iop]: self.nnops[iop + 1]])
                    for iop, oper in enumerate(self.ops)
                ],
            )
            y1 = np.array(list(y1))
        val = comm.scatter(y1, root=0)
        return comm.allreduce(val, op=MPI.SUM)
