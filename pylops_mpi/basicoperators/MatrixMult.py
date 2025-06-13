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


class MPIMatrixMult(MPILinearOperator):
    def __init__(
            self,
            A: NDArray,
            N: int,
            saveAt: bool = False,
            base_comm: MPI.Comm = MPI.COMM_WORLD,
            dtype: DTypeLike = "float64",
    ) -> None:
        rank = base_comm.Get_rank()
        size = base_comm.Get_size()

        # Determine grid dimensions (P_prime × C) such that P_prime * C ≥ size
        self._P_prime = int(math.ceil(math.sqrt(size)))
        self._C = int(math.ceil(size / self._P_prime))
        if self._P_prime * self._C != size:
            raise Exception("Number of Procs must be a square number")

        # Compute this process's group and layer indices
        self._group_id = rank % self._P_prime
        self._layer_id = rank // self._P_prime

        # Split communicators by layer (rows) and by group (columns)
        self.base_comm = base_comm
        self._layer_comm = base_comm.Split(color=self._layer_id, key=self._group_id)
        self._group_comm = base_comm.Split(color=self._group_id, key=self._layer_id)
        self.A = A.astype(np.dtype(dtype))
        if saveAt: self.At = A.T.conj()

        self.M = self._layer_comm.allreduce(self.A.shape[0], op=MPI.SUM)
        self.K = A.shape[1]
        self.N = N

        # Determine how many columns each group holds
        block_cols = int(math.ceil(self.N / self._P_prime))
        blk_rows = int(math.ceil(self.M / self._P_prime))

        self._row_start = self._group_id * blk_rows
        self._row_end = min(self.M, self._row_start + blk_rows)

        self._col_start = self._layer_id * block_cols
        self._col_end = min(self.N, self._col_start + block_cols)

        self._local_ncols = self._col_end - self._col_start
        self._rank_col_lens = self.base_comm.allgather(self._local_ncols)
        total_ncols = np.sum(self._rank_col_lens)

        self.dims = (self.K, total_ncols)
        self.dimsd = (self.M, total_ncols)
        shape = (int(np.prod(self.dimsd)), int(np.prod(self.dims)))
        super().__init__(shape=shape, dtype=np.dtype(dtype), base_comm=base_comm)

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER} Got {x.partition} instead...")

        y = DistributedArray(global_shape=(self.M * self.dimsd[1]),
                             local_shapes=[(self.M * c) for c in self._rank_col_lens],
                             mask=x.mask,
                             partition=Partition.SCATTER,
                             dtype=self.dtype)

        my_own_cols = self._rank_col_lens[self.rank]
        x_arr = x.local_array.reshape((self.dims[0], my_own_cols))
        x_arr = x_arr.astype(self.dtype)

        X_local = self._layer_comm.bcast(x_arr if self._group_id == self._layer_id else None, root=self._layer_id)
        Y_local = ncp.vstack(
            self._layer_comm.allgather(
                ncp.matmul(self.A, X_local)
            )
        )
        y[:] = Y_local.flatten()
        return y

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER}. Got {x.partition} instead.")

        y = DistributedArray(
            global_shape=(self.K * self.dimsd[1]),
            local_shapes=[self.K * c for c in self._rank_col_lens],
            mask=x.mask,
            partition=Partition.SCATTER,
            dtype=self.dtype,
        )

        x_arr = x.local_array.reshape((self.M, self._local_ncols)).astype(self.dtype)
        X_tile = x_arr[self._row_start:self._row_end, :]

        A_local = self.At if hasattr(self, "At") else self.A.T.conj()
        m, b = A_local.shape
        pad = (-m) % self._P_prime
        A_pad = A_local if pad <= 0 else np.pad(A_local, ((0, pad), (0, 0)), mode='constant', constant_values=self.dtype.type(0.0))
        batch_sz = (m + pad) // self._P_prime
        A_batch  = A_pad.reshape(self._P_prime, batch_sz, b)

        Y_batch = ncp.matmul(A_batch, X_tile)
        Y_pad = Y_batch.reshape(batch_sz * self._P_prime, -1)
        y_local = Y_pad[:A_local.shape[0], :]
        y_layer = self._layer_comm.allreduce(y_local, op=MPI.SUM)
        y[:] = y_layer.flatten()
        return y
