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
    r"""MPI Matrix multiplication
    
    Implement distributed matrix-matrix multiplication between a matrix 
    :math:`\mathbf{A}` blocked over rows (i.e., blocks of rows are stored
    over different ranks) and the input model and data vector, which are both to
    be interpreted as matrices blocked over columns.

    Parameters
    ----------
    A : :obj:`numpy.ndarray`
        Local block of the matrix of shape :math:`[M_{loc} \times K]`
        where ``M_loc`` is the number of rows stored on this MPI rank and
        ``K`` is the global number of columns.
    N : :obj:`int`
        Global leading dimension (i.e., number of columns) of the matrices 
        representing the input model and data vectors.
    saveAt : :obj:`bool`, optional
        Save ``A`` and ``A.H`` to speed up the computation of adjoint
        (``True``) or create ``A.H`` on-the-fly (``False``)
        Note that ``saveAt=True`` will double the amount of required memory.
        Default is ``False``.
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        MPI Base Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape

    Raises
    ------
    Exception
        If the operator is created without a square number of mpi ranks.
    ValueError
        If input vector does not have the correct partition type.

    Notes
    -----
    This operator performs a matrix-matrix multiplication, whose forward 
    operation can be described as :math:`Y = A \cdot X` where:

    - :math:`\mathbf{A}` is the distributed matrix operator of shape :math:`[M \times K]`
    - :math:`\mathbf{X}` is the distributed operand matrix of shape :math:`[K \times N]`
    - :math:`\mathbf{Y}` is the resulting distributed matrix of shape :math:`[M \times N]`

    whilst the adjoint operation is represented by 
    :math:`\mathbf{X}_{adj} = \mathbf{A}^H \cdot \mathbf{Y}` where 
    :math:`\mathbf{A}^H` is the complex conjugate and transpose of :math:`\mathbf{A}`.
    
    This implementation is based on a 1D block distribution of the operator 
    matrix and reshaped model and data vectors replicated across math:`P` 
    processes by a factor equivalent to :math:`\sqrt{P}` across a square process 
    grid (:math:`\sqrt{P}\times\sqrt{P}`). More specifically:

    - The matrix ``A`` is distributed across MPI processes in a block-row fashion
      and each process holds a local block of ``A`` with shape 
      :math:`[M_{loc} \times K]`
    - The operand matrix ``X`` is distributed in a block-column fashion and
      and each process holds a local block of ``X`` with shape 
      :math:`[K \times N_{loc}]`
    - Communication is minimized by using a 2D process grid layout

    **Forward Operation step-by-step**

    1. **Input Preparation**: The input vector ``x`` (flattened from matrix ``X``
       of shape ``(K, N)``) is reshaped to ``(K, N_local)`` where ``N_local``
       is the number of columns assigned to the current process.

    2. **Data Broadcasting**: Within each layer (processes with same ``layer_id``),
       the operand data is broadcast from the process whose ``group_id`` matches
       the ``layer_id``. This ensures all processes in a layer have access to
       the same operand columns.

    3. **Local Computation**: Each process computes ``A_local @ X_local`` where:
       - ``A_local`` is the local block of matrix ``A`` (shape ``M_local x K``)
       - ``X_local`` is the broadcasted operand (shape ``K x N_local``)

    4. **Layer Gather**: Results from all processes in each layer are gathered
       using ``allgather`` to reconstruct the full result matrix vertically.

    **Adjoint Operation step-by-step**

    The adjoint operation performs the conjugate transpose multiplication:

    1. **Input Reshaping**: The input vector ``x`` is reshaped to ``(M, N_local)``
       representing the local columns of the input matrix.

    2. **Local Adjoint Computation**:
        Each process computes ``A_local.H @ X_tile``
            where ``A_local.H`` is either:
                - Pre-computed ``At`` (if ``saveAt=True``)
                - Computed on-the-fly as ``A.T.conj()`` (if ``saveAt=False``)
        Each process multiplies its transposed  local ``A`` block ``A_local^H`` 
        (shape ``K x M_block``)
        with the extracted  ``X_tile`` (shape ``M_block x N_local``),
        producing a partial result of  shape ``(K, N_local)``.
        This computes the local contribution of columns of  ``A^H`` to the final result.

    3. **Layer Reduction**: Since the full result ``Y = A^H \cdot X`` is the
       sum of contributions from all column blocks of ``A^H``, processes in the 
       same layer perform an ``allreduce`` sum to combine their partial results. 
       This gives the complete ``(K, N_local)`` result for their assigned columns.
    
    """
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
            raise Exception(f"Number of processes must be a square number, provided {size} instead...")

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
        Y_local = ncp.matmul(A_local, X_tile)
        y_layer = self._layer_comm.allreduce(Y_local, op=MPI.SUM)
        y[:] = y_layer.flatten()
        return y
