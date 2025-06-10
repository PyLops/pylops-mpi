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


class MPISUMMAMatrixMult(MPILinearOperator):
    def __init__(
            self,
            A: NDArray,
            N: int,
            base_comm: MPI.Comm = MPI.COMM_WORLD,
            dtype: DTypeLike = "float64",
    ) -> None:
        rank = base_comm.Get_rank()
        size = base_comm.Get_size()

        # Determine grid dimensions (P_prime × C) such that P_prime * C ≥ size
        self._P_prime = int(math.ceil(math.sqrt(size)))
        self._C = int(math.ceil(size / self._P_prime))
        assert self._P_prime * self._C >= size

        # Compute this process's group and layer indices
        self._group_id = rank % self._P_prime
        self._layer_id = rank // self._P_prime

        # Split communicators by layer (rows) and by group (columns)
        self.base_comm   = base_comm
        self._layer_comm = base_comm.Split(color=self._layer_id, key=self._group_id)
        self._group_comm = base_comm.Split(color=self._group_id, key=self._layer_id)

        self.dtype = np.dtype(dtype)
        self.A    = np.array(A, dtype=self.dtype, copy=False)

        self.M = self._layer_comm.allreduce(self.A.shape[0], op=MPI.SUM)
        self.K = A.shape[1]
        self.N = N

        # Determine how many columns each group holds
        block_cols = int(math.ceil(self.N / self._P_prime))
        local_col_start = self._group_id * block_cols
        local_col_end = min(self.N, local_col_start + block_cols)
        local_ncols = local_col_end - local_col_start

        # Sum up the total number of input columns across all processes
        total_ncols = base_comm.allreduce(local_ncols, op=MPI.SUM)
        self.dims = (self.K, total_ncols)

        # Recompute how many output columns each layer holds
        layer_col_start  = self._layer_id * block_cols
        layer_col_end    = min(self.N, layer_col_start + block_cols)
        layer_ncols      = layer_col_end - layer_col_start
        total_layer_cols = self.base_comm.allreduce(layer_ncols, op=MPI.SUM)

        self.dimsd = (self.M, total_layer_cols)
        shape = (int(np.prod(self.dimsd)), int(np.prod(self.dims)))

        super().__init__(shape=shape, dtype=np.dtype(dtype), base_comm=base_comm)
        
    def _matvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER} Got {x.partition} instead...")
        blk_cols    = int(math.ceil(self.N / self._P_prime))
        col_start   = self._group_id * blk_cols
        col_end     = min(self.N, col_start + blk_cols)
        my_own_cols = max(0, col_end - col_start)
        x = x.local_array.reshape((self.dims[0], my_own_cols))
        x = x.astype(self.dtype, copy=False)
        B_block = self._layer_comm.bcast(x if self._group_id == self._layer_id else None, root=self._layer_id)
        C_local = ncp.vstack(
            self._layer_comm.allgather(
                ncp.matmul(self.A, B_block)
            )
        )

        layer_col_start = self._layer_id * blk_cols
        layer_col_end   = min(self.N, layer_col_start + blk_cols)
        layer_ncols     = max(0, layer_col_end - layer_col_start)
        layer_col_lens  = self.base_comm.allgather(layer_ncols)
        mask = [i // self._P_prime for i in range(self.size)]

        y = DistributedArray(global_shape= (self.M * self.dimsd[1]),
                             local_shapes=[(self.M * c) for c in layer_col_lens],
                             mask=mask,
                             partition=Partition.SCATTER,
                             dtype=self.dtype)
        y[:] = C_local.flatten()
        return y

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER}. Got {x.partition} instead.")

        # Determine local column block for this layer
        blk_cols        = int(math.ceil(self.N / self._P_prime))
        layer_col_start = self._layer_id * blk_cols
        layer_col_end   = min(self.N, layer_col_start + blk_cols)
        layer_ncols     = layer_col_end - layer_col_start
        layer_col_lens  = self.base_comm.allgather(layer_ncols)
        x               = x.local_array.reshape((self.M, layer_ncols))

        # Determine local row block for this process group
        blk_rows  = int(math.ceil(self.M / self._P_prime))
        row_start = self._group_id * blk_rows
        row_end   = min(self.M, row_start + blk_rows)

        B_tile = x[row_start:row_end, :].astype(self.dtype, copy=False)
        A_local = self.A.T.conj()

        m, b    = A_local.shape
        pad     = (-m) % self._P_prime
        r       = (m + pad) // self._P_prime
        A_pad   = np.pad(A_local, ((0, pad), (0, 0)),  mode='constant', constant_values=0)
        A_batch = A_pad.reshape(self._P_prime, r, b)

        # Perform local matmul and unpad
        Y_batch = ncp.matmul(A_batch, B_tile)
        Y_pad   = Y_batch.reshape(r * self._P_prime, -1)
        y_local = Y_pad[:m, :]
        y_layer = self._layer_comm.allreduce(y_local, op=MPI.SUM)

        mask = [i // self._P_prime for i in range(self.size)]
        y = DistributedArray(
            global_shape=(self.K * self.dimsd[1]),
            local_shapes=[self.K * c for c in layer_col_lens],
            mask=mask,
            partition=Partition.SCATTER,
            dtype=self.dtype,
        )
        y[:] = y_layer.flatten()
        return y
