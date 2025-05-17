import numpy as np
import math
from mpi4py import MPI
from pylops.utils.backend import get_module
from pylops.utils.typing import DTypeLike, NDArray

from pylops_mpi import (
    DistributedArray,
    MPILinearOperator,
    Partition
)

class SUMMAMatrixMult(MPILinearOperator):
    def __init__(
            self,
            A: NDArray, #I am going to have to assume that the partitioning has been done correctly
            N: int,
            base_comm: MPI.Comm = MPI.COMM_WORLD,
            dtype: DTypeLike = "float64",
    ) -> None:
        rank   = base_comm.Get_rank()
        nProcs = base_comm.Get_size()
        self._P_prime = int(math.ceil(math.sqrt(nProcs)))
        self._C = int(math.ceil(nProcs / self._P_prime))
        assert self._P_prime * self._C >= nProcs

        self.N = N
        self.A = A
        self._my_group   = rank % self._P_prime
        self._my_layer   = rank // self._P_prime
        self._layer_comm = base_comm.Split(color=self._my_layer, key=self._my_group)
        self._group_comm = base_comm.Split(color=self._my_group, key=self._my_layer)
        K_global = A.shape[1]
        if N % self._P_prime != 0:
            raise ValueError(f"N should be divisible by sqrt(Processes)")
        blk_cols = int(math.ceil(N / self._P_prime))
        self.dims = (nProcs, K_global, blk_cols)
        super().__init__(shape=(1, int(np.prod(self.dims))), dtype=np.dtype(dtype), base_comm=base_comm)

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER} Got {x.partition} instead...")
        blk_cols     = int(math.ceil(self.N / self._P_prime))
        col_start    = self._my_group * blk_cols
        col_end      = min(self.N, col_start + blk_cols)
        my_own_cols  = col_end - col_start
        x = x.local_array.reshape((self.dims[1], my_own_cols))
        C_local = None
        for t in range(self._P_prime):
            responsible_layer = t % self._C
            if self._my_layer == responsible_layer:
                B_block = self._layer_comm.bcast(x if self._my_group == t else None, root=t)
                if t == self._my_layer: C_local = ncp.matmul(self.A, B_block)
        self.base_comm.Barrier()
        my_C_rows    = ncp.hstack(self._group_comm.allgather(C_local))

        M_global = self._layer_comm.allreduce(self.A.shape[0], op=MPI.SUM)
        blk_rows = int(math.ceil(M_global / self._P_prime))
        mask = [i % self._P_prime for i in range(self.base_comm.Get_size())]
        y    = DistributedArray(global_shape=(self.size, blk_rows , self.N),
                                mask = mask,
                                partition=Partition.SCATTER)
        y[:] = my_C_rows
        return y