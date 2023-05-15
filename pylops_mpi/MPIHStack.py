from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import numpy as np
from pylops import LinearOperator, MatrixMult
from scipy.sparse.linalg import LinearOperator as spLinearOperator
from pylops.utils import NDArray

comm = MPI.COMM_WORLD


def _matvec_rmatvec_map(op, x: NDArray) -> NDArray:
    return op(x).squeeze()


class MPIHStack:

    def __init__(self, ops, max_workers: int = 1) -> None:
        self.ops = ops
        mops = np.zeros(len(ops), dtype=int)
        for iop, oper in enumerate(ops):
            if not isinstance(oper, (LinearOperator, spLinearOperator)):
                self.ops[iop] = MatrixMult(oper, dtype=oper.dtype)
            mops[iop] = self.ops[iop].shape[1]
        self.mops = int(mops.sum())
        nops = [oper.shape[0] for oper in self.ops]
        if len(set(nops)) > 1:
            raise ValueError("operators have different number of rows")
        self.nops = int(nops[0])
        self.mmops = np.insert(np.cumsum(mops), 0, 0)
        # create MPI pooL
        self._max_workers = max_workers
        self.mpiexec = None
        if self._max_workers > 1:
            self.mpiexec = MPIPoolExecutor(max_workers=max_workers)

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @max_workers.setter
    def max_workers(self, new_workers: int):
        if self.max_workers > 1:
            self.mpiexec.shutdown()
        if new_workers > 1:
            self.mpiexec = MPIPoolExecutor(processes=new_workers)
        self._max_workers = new_workers

    def _matvec_multiproc(self, x: NDArray) -> NDArray:
        y1 = None
        if comm.Get_rank() == 0:
            y1 = self.mpiexec.starmap(
                _matvec_rmatvec_map,
                [
                    (oper._matvec, x[self.mmops[iop]: self.mmops[iop + 1]])
                    for iop, oper in enumerate(self.ops)
                ],
            )
            y1 = np.array(list(y1))
        val = comm.scatter(y1, root=0)
        return comm.allreduce(val, op=MPI.SUM)
