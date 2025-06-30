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
        Local block of the matrix of shape :math:`[N_{loc} \times K]`
        where :math:`N_{loc}` is the number of rows stored on this MPI rank and
        ``K`` is the global number of columns.
    M : :obj:`int`
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
        If the operator is created with a non-square number of MPI ranks.
    ValueError
        If input vector does not have the correct partition type.

    Notes
    -----
    This operator performs a matrix-matrix multiplication, whose forward
    operation can be described as :math:`Y = A \cdot X` where:

    - :math:`\mathbf{A}` is the distributed matrix operator of shape :math:`[N \times K]`
    - :math:`\mathbf{X}` is the distributed operand matrix of shape :math:`[K \times M]`
    - :math:`\mathbf{Y}` is the resulting distributed matrix of shape :math:`[N \times M]`

    whilst the adjoint operation is represented by
    :math:`\mathbf{X}_{adj} = \mathbf{A}^H \cdot \mathbf{Y}` where
    :math:`\mathbf{A}^H` is the complex conjugate and transpose of :math:`\mathbf{A}`.

    This implementation is based on a 1D block distribution of the operator
    matrix and reshaped model and data vectors replicated across :math:`P`
    processes by a factor equivalent to :math:`\sqrt{P}` across a square process
    grid (:math:`\sqrt{P}\times\sqrt{P}`). More specifically:

    - The matrix ``A`` is distributed across MPI processes in a block-row fashion
      and each process holds a local block of ``A`` with shape
      :math:`[N_{loc} \times K]`
    - The operand matrix ``X`` is distributed in a block-column fashion and
      each process holds a local block of ``X`` with shape
      :math:`[K \times M_{loc}]`
    - Communication is minimized by using a 2D process grid layout

    **Forward Operation step-by-step**

    1. **Input Preparation**: The input vector ``x`` (flattened from matrix ``X``
       of shape ``(K, M)``) is reshaped to ``(K, M_local)`` where ``M_local``
       is the number of columns assigned to the current process.

    2. **Local Computation**: Each process computes ``A_local @ X_local`` where:
       - ``A_local`` is the local block of matrix ``A`` (shape ``N_local x K``)
       - ``X_local`` is the broadcasted operand (shape ``K x M_local``)

    3. **Row-wise Gather**: Results from all processes in each row are gathered
       using ``allgather`` to ensure that each rank has a block-column of the
       output matrix.

    **Adjoint Operation step-by-step**

    The adjoint operation performs the conjugate transpose multiplication:

    1. **Input Reshaping**: The input vector ``x`` is reshaped to ``(N, M_local)``
       representing the local columns of the input matrix.

    2. **Local Adjoint Computation**: Each process computes
       ``A_local.H @ X_tile`` where ``A_local.H`` is either i) Pre-computed
       and stored in ``At`` (if ``saveAt=True``), ii) computed on-the-fly as
       ``A.T.conj()`` (if ``saveAt=False``). Each process multiplies its
       transposed  local ``A`` block ``A_local^H`` (shape ``K x N_block``)
       with the extracted  ``X_tile`` (shape ``N_block x M_local``),
       producing a partial result of  shape ``(K, M_local)``.
       This computes the local contribution of columns of  ``A^H`` to the final
       result.

    3. **Row-wise Reduction**: Since the full result ``Y = A^H \cdot X`` is the
       sum of the contributions from all column blocks of ``A^H``, processes in
       the same row perform an ``allreduce`` sum to combine their partial results.
       This gives the complete ``(K, M_local)`` result for their assigned column.

    """
    def __init__(
            self,
            A: NDArray,
            M: int,
            saveAt: bool = False,
            base_comm: MPI.Comm = MPI.COMM_WORLD,
            dtype: DTypeLike = "float64",
    ) -> None:
        rank = base_comm.Get_rank()
        size = base_comm.Get_size()

        # Determine grid dimensions (P_prime × C) such that P_prime * C ≥ size
        self._P_prime = math.isqrt(size)
        self._C = self._P_prime
        if self._P_prime * self._C != size:
            raise Exception(f"Number of processes must be a square number, provided {size} instead...")

        self._col_id = rank % self._P_prime
        self._row_id = rank // self._P_prime

        self.base_comm = base_comm
        self._row_comm = base_comm.Split(color=self._row_id, key=self._col_id)
        self._col_comm = base_comm.Split(color=self._col_id, key=self._row_id)

        self.A = A.astype(np.dtype(dtype))
        if saveAt:
            self.At = A.T.conj()

        self.N = self._row_comm.allreduce(self.A.shape[0], op=MPI.SUM)
        self.K = A.shape[1]
        self.M = M

        block_cols = int(math.ceil(self.M / self._P_prime))
        blk_rows = int(math.ceil(self.N / self._P_prime))

        self._row_start = self._col_id * blk_rows
        self._row_end = min(self.N, self._row_start + blk_rows)

        self._col_start = self._row_id * block_cols
        self._col_end = min(self.M, self._col_start + block_cols)

        self._local_ncols = max(0, self._col_end - self._col_start)
        self._rank_col_lens = self.base_comm.allgather(self._local_ncols)
        total_ncols = np.sum(self._rank_col_lens)

        self.dims = (self.K, total_ncols)
        self.dimsd = (self.N, total_ncols)
        shape = (int(np.prod(self.dimsd)), int(np.prod(self.dims)))
        super().__init__(shape=shape, dtype=np.dtype(dtype), base_comm=base_comm)

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER} Got {x.partition} instead...")

        y = DistributedArray(
            global_shape=(self.N * self.dimsd[1]),
            local_shapes=[(self.N * c) for c in self._rank_col_lens],
            mask=x.mask,
            partition=Partition.SCATTER,
            dtype=self.dtype,
            base_comm=self.base_comm
        )

        my_own_cols = self._rank_col_lens[self.rank]
        x_arr = x.local_array.reshape((self.dims[0], my_own_cols))
        X_local = x_arr.astype(self.dtype)
        Y_local = ncp.vstack(
            self._row_comm.allgather(
                ncp.matmul(self.A, X_local)
            )
        )
        y[:] = Y_local.flatten()
        return y

    @staticmethod
    def active_grid_comm(base_comm:MPI.Comm, N:int, M:int):
        """
        Configure a square process grid from a parent MPI communicator and select the subset of "active" processes.

        Each process in base_comm is assigned to a logical 2D grid of size p_prime x p_prime,
        where p_prime = floor(sqrt(total_ranks)). Only the first `active_dim x active_dim` processes
        (by row-major order) are considered "active". Inactive ranks return immediately with no new communicator.

        Parameters:
        -----------
        base_comm : MPI.Comm
            The parent communicator (e.g., MPI.COMM_WORLD).
        N : int
            Number of rows of your global data domain.
        M : int
            Number of columns of your global data domain.

        Returns:
        --------
        tuple:
            comm (MPI.Comm or None) : Sub-communicator including only active ranks.
            rank (int)              : Rank within the new sub-communicator (or original rank if inactive).
            row (int)               : Grid row index of this process in the active grid (or original rank if inactive).
            col (int)               : Grid column index of this process in the active grid (or original rank if inactive).
            is_active (bool)        : Flag indicating whether this rank is in the active sub-grid.
        """
        rank = base_comm.Get_rank()
        size = base_comm.Get_size()
        p_prime = math.isqrt(size)
        row, col = divmod(rank, p_prime)
        active_dim = min(N, M, p_prime)
        is_active = (row < active_dim and col < active_dim)

        if not is_active:
            return None, rank, row, col, False

        active_ranks = [r for r in range(size)
                        if (r // p_prime) < active_dim and (r % p_prime) < active_dim]
        new_group = base_comm.Get_group().Incl(active_ranks)
        new_comm = base_comm.Create_group(new_group)

        p_prime_new = math.isqrt(len(active_ranks))
        new_rank = new_comm.Get_rank()
        new_row, new_col = divmod(new_rank, p_prime_new)
        return new_comm, new_rank, new_row, new_col, True

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
            base_comm=self.base_comm
        )

        x_arr = x.local_array.reshape((self.N, self._local_ncols)).astype(self.dtype)
        X_tile = x_arr[self._row_start:self._row_end, :]
        A_local = self.At if hasattr(self, "At") else self.A.T.conj()
        Y_local = ncp.matmul(A_local, X_tile)
        y_layer = self._row_comm.allreduce(Y_local, op=MPI.SUM)
        y[:] = y_layer.flatten()
        return y
