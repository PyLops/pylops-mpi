import math
import numpy as np
from typing import Tuple
from mpi4py import MPI

from pylops.utils.backend import get_module
from pylops.utils.typing import DTypeLike, NDArray

from pylops_mpi import (
    DistributedArray,
    MPILinearOperator,
    Partition
)


def active_grid_comm(base_comm: MPI.Comm, N: int, M: int):
    r"""Configure active grid

    Configure a square process grid from a parent MPI communicator and
    select a subset of "active" processes. Each process in ``base_comm``
    is assigned to a logical 2D grid of size :math:`P' \times P'`,
    where :math:`P' = \bigl \lceil \sqrt{P} \bigr \rceil`. Only the first
    :math:`active_dim x active_dim` processes
    (by row-major order) are considered "active". Inactive ranks return
    immediately with no new communicator.

    Parameters:
    -----------
    base_comm : :obj:`mpi4py.MPI.Comm`
        MPI Parent Communicator. (e.g., ``mpi4py.MPI.COMM_WORLD``).
    N : :obj:`int`
        Number of rows of the global data domain.
    M : :obj:`int`
        Number of columns of the global data domain.

    Returns:
    --------
    comm : :obj:`mpi4py.MPI.Comm`
        Sub-communicator including only active ranks.
    rank : :obj:`int`
        Rank within the new sub-communicator (or original rank
        if inactive).
    row : :obj:`int`
        Grid row index of this process in the active grid (or original rank
        if inactive).
    col : :obj:`int`
        Grid column index of this process in the active grid
        (or original rank if inactive).
    is_active : :obj:`bool`
        Flag indicating whether this rank is in the active sub-grid.

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


def local_block_spit(global_shape: Tuple[int, int],
                     rank: int,
                     comm: MPI.Comm) -> Tuple[slice, slice]:
    """
    Compute the local sub‐block of a 2D global array for a process in a square process grid.

    Parameters
    ----------
    global_shape : Tuple[int, int]
        Dimensions of the global 2D array (n_rows, n_cols).
    rank : int
        Rank of the MPI process in `comm` for which to get the owned block partition.
    comm : MPI.Comm
        MPI communicator whose total number of processes :math:`\mathbf{P}`
        must be a perfect square :math:`\mathbf{P} = \sqrt{\mathbf{P'}}`.

    Returns
    -------
    Tuple[slice, slice]
        Two `slice` objects `(row_slice, col_slice)` indicating the sub‐block
        of the global array owned by this rank.

    Raises
    ------
    ValueError
        if `rank` is out of range.
    RuntimeError
        If the number of processes participating in the provided communicator is not a perfect square.
    """
    size = comm.Get_size()
    p_prime = math.isqrt(size)
    if p_prime * p_prime != size:
        raise RuntimeError(f"Number of processes must be a square number, provided {size} instead...")
    if not ( isinstance(rank, int) and 0 <= rank < size ):
        raise ValueError(f"rank must be integer in [0, {size}), got {rank!r}")

    proc_i, proc_j = divmod(rank, p_prime)
    orig_r, orig_c = global_shape

    new_r = math.ceil(orig_r / p_prime) * p_prime
    new_c = math.ceil(orig_c / p_prime) * p_prime

    blkr, blkc = new_r // p_prime, new_c // p_prime

    i0, j0 = proc_i * blkr, proc_j * blkc
    i1, j1 = min(i0 + blkr, orig_r), min(j0 + blkc, orig_c)

    return slice(i0, i1), slice(j0, j1)


def block_gather(x: DistributedArray, new_shape: Tuple[int, int], orig_shape: Tuple[int, int], comm: MPI.Comm):
    """
    Gather distributed local blocks from 2D block distributed matrix distributed
    amongst a square process grid into the full global array.

    Parameters
    ----------
    x : :obj:`pylops_mpi.DistributedArray`
        The distributed array to gather locally.
    new_shape : Tuple[int, int]
        Shape `(N', M')` of the padded global array, where both dimensions
        are multiples of :math:`\sqrt{\mathbf{P}}`.
    orig_shape : Tuple[int, int]
        Original shape `(N, M)` of the global array before padding.
    comm : MPI.Comm
        MPI communicator whose size must be a perfect square (P = p_prime**2).

    Returns
    -------
    Array
        The reconstructed 2D array of shape `orig_shape`, assembled from
        the distributed blocks.

    Raises
    ------
    RuntimeError
        If the number of processes participating in the provided communicator is not a perfect square.
    """
    ncp = get_module(x.engine)
    p_prime = math.isqrt(comm.Get_size())
    if p_prime * p_prime != comm.Get_size():
        raise RuntimeError(f"Communicator size must be a perfect square, got {comm.Get_size()!r}")

    all_blks = comm.allgather(x.local_array)

    nr, nc = new_shape
    orr, orc = orig_shape

    # Calculate base block sizes
    br_base = nr // p_prime
    bc_base = nc // p_prime

    # Calculate remainder rows/cols that need to be distributed
    r_remainder = nr % p_prime
    c_remainder = nc % p_prime

    # Create the output matrix
    C = ncp.zeros((nr, nc), dtype=all_blks[0].dtype)

    # Place each block in the correct position
    for rank in range(p_prime * p_prime):
        # Convert linear rank to 2D grid position
        proc_row = rank // p_prime
        proc_col = rank % p_prime

        # Calculate this process's block dimensions
        block_rows = br_base + (1 if proc_row < r_remainder else 0)
        block_cols = bc_base + (1 if proc_col < c_remainder else 0)

        # Calculate starting position in global matrix
        start_row = proc_row * br_base + min(proc_row, r_remainder)
        start_col = proc_col * bc_base + min(proc_col, c_remainder)

        # Place the block
        block = all_blks[rank]
        if block.ndim == 1:
            block = block.reshape(block_rows, block_cols)
        C[start_row:start_row + block_rows,
          start_col:start_col + block_cols] = block

    # Trim off any padding
    return C[:orr, :orc]



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

class MPISummaMatrixMult(MPILinearOperator):
    r"""MPI SUMMA Matrix multiplication

    Implements distributed matrix-matrix multiplication using the SUMMA algorithm
    between a matrix :math:`\mathbf{A}` distributed over a 2D process grid and
    input model and data vectors, which are both interpreted as matrices
    distributed in block fashion wherein each process owns a tile of the matrix.

    Parameters
    ----------
    A : :obj:`numpy.ndarray`
        Local block of the matrix of shape :math:`[N_{loc} \times K_{loc}]`
        where :math:`N_{loc}` and :math:`K_{loc}` are the number of rows and
        columns stored on this MPI rank.
    M : :obj:`int`
        Global number of columns of the matrices representing the input model
        and data vectors.
    saveAt : :obj:`bool`, optional
        Save ``A`` and ``A.H`` to speed up the computation of adjoint
        (``True``) or create ``A.H`` on-the-fly (``False``).
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
    This operator performs distributed matrix-matrix multiplication using the
    SUMMA (Scalable Universal Matrix Multiplication Algorithm), whose forward
    operation can be described as :math:`\mathbf{Y} = \mathbf{A} \cdot \mathbf{X}` where:

    - :math:`\mathbf{A}` is the distributed matrix operator of shape :math:`[N \times K]`
    - :math:`\mathbf{X}` is the distributed operand matrix of shape :math:`[K \times M]`
    - :math:`\mathbf{Y}` is the resulting distributed matrix of shape :math:`[N \times M]`

    The adjoint operation is represented by
    :math:`\mathbf{X}_{adj} = \mathbf{A}^H \cdot \mathbf{Y}` where
    :math:`\mathbf{A}^H` is the complex conjugate transpose of :math:`\mathbf{A}`.

    This implementation is based on a 2D block distribution across a square process
    grid (:math:`\sqrt{P}\times\sqrt{P}`). The matrices are distributed as follows:

    - The matrix ``A`` is distributed across MPI processes in 2D blocks where
      each process holds a local block of ``A`` with shape :math:`[N_{loc} \times K_{loc}]`
      where :math:`N_{loc} = \frac{N}{\sqrt{P}}` and :math:`K_{loc} = \frac{K}{\sqrt{P}}`.

    - The operand matrix ``X`` is also distributed across MPI processes in 2D blocks where
      each process holds a local block of ``X`` with shape :math:`[K_{loc} \times M_{loc}]`
      where :math:`K_{loc} = \frac{K}{\sqrt{P}}` and :math:`M_{loc} = \frac{M}{\sqrt{P}}`.

    - The result matrix ``Y`` is also distributed across MPI processes in 2D blocks where
      each process holds a local block of ``Y`` with shape :math:`[N_{loc} \times M_{loc}]`
      where :math:`N_{loc} = \frac{N}{\sqrt{P}}` and :math:`M_{loc} = \frac{M}{\sqrt{P}}`.


    **Forward Operation (SUMMA Algorithm)**

    The forward operation implements the SUMMA algorithm:

    1. **Input Preparation**: The input vector ``x``is reshaped to ``(K_{loc}, M_{loc})`` representing
       the local block assigned to the current process.

    2. **SUMMA Iteration**: For each step ``k`` in the SUMMA algorithm  -- :math:`k \in \[ 0, \sqrt{P} \)}` :

       a. **Broadcast A blocks**: Process in column ``k`` broadcasts its ``A``
          block to all other processes in the same process row.

       b. **Broadcast X blocks**: Process in row ``k`` broadcasts its ``X``
          block to all other processes in the same process column.

       c. **Local Computation**: Each process computes the partial matrix
          product ``A_broadcast @ X_broadcast`` and accumulates it to its
          local result.

    3. **Result Assembly**: After all k SUMMA iterations, each process has computed
       its local block of the result matrix ``Y``.

    **Adjoint Operation (SUMMA Algorithm)**

    The adjoint operation performs the conjugate transpose multiplication using
    a modified SUMMA algorithm:

    1. **Input Reshaping**: The input vector ``x`` is reshaped to ``(N_{loc}, M_{loc})``
       representing the local block of the input matrix.

    2. **SUMMA Adjoint Iteration**: For each step ``k`` in the adjoint SUMMA algorithm:

       a. **Broadcast A^H blocks**: The conjugate transpose of ``A`` blocks is
          communicated between processes. If ``saveAt=True``, the pre-computed
          ``A.H`` is used; otherwise, ``A.T.conj()`` is computed on-the-fly.

       b. **Broadcast Y blocks**: Process in row ``k`` broadcasts its ``Y``
          block to all other processes in the same process column.

       c. **Local Adjoint Computation**: Each process computes the partial
          matrix product ``A_H_broadcast @ Y_broadcast`` and accumulates it
          to the local result.

    3. **Result Assembly**: After all adjoint SUMMA iterations, each process has
       computed its local block of the result matrix ``X_{adj}``.

    The implementation handles padding automatically to ensure proper block sizes
    for the square process grid, and unpadding is performed before returning results.

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
        self._P_prime =  math.isqrt(size)
        if self._P_prime * self._P_prime != size:
            raise Exception(f"Number of processes must be a square number, provided {size} instead...")

        self._row_id, self._col_id =  divmod(rank, self._P_prime)

        self.base_comm = base_comm
        self._row_comm = base_comm.Split(color=self._row_id, key=self._col_id)
        self._col_comm = base_comm.Split(color=self._col_id, key=self._row_id)

        self.A = A.astype(np.dtype(dtype))
        if saveAt: self.At = A.T.conj()

        self.N = self._col_comm.allreduce(A.shape[0])
        self.K = self._row_comm.allreduce(A.shape[1])
        self.M = M

        self._N_padded = math.ceil(self.N / self._P_prime) * self._P_prime
        self._K_padded = math.ceil(self.K / self._P_prime) * self._P_prime
        self._M_padded = math.ceil(self.M / self._P_prime) * self._P_prime

        bn = self._N_padded // self._P_prime
        bk = self._K_padded // self._P_prime
        bm = self._M_padded // self._P_prime

        pr = (bn - A.shape[0]) if self._row_id == self._P_prime - 1 else 0
        pc = (bk - A.shape[1]) if self._col_id == self._P_prime - 1 else 0

        if pr > 0 or pc > 0:
            self.A = np.pad(self.A, [(0, pr), (0, pc)], mode='constant')

        if saveAt:
            self.At = self.A.T.conj()

        self.dims  = (self.K, self.M)
        self.dimsd = (self.N, self.M)
        shape = (int(np.prod(self.dimsd)), int(np.prod(self.dims)))
        super().__init__(shape=shape, dtype=np.dtype(dtype), base_comm=base_comm)

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER} Got {x.partition} instead...")

        # Calculate local shapes for block distribution
        bn = self._N_padded // self._P_prime  # block size in N dimension
        bm = self._M_padded // self._P_prime  # block size in M dimension

        local_n = bn if self._row_id != self._P_prime - 1 else self.N - (self._P_prime - 1) * bn
        local_m = bm if self._col_id != self._P_prime - 1 else self.M - (self._P_prime - 1) * bm

        local_shapes = self.base_comm.allgather(local_n * local_m)

        y = DistributedArray(global_shape=(self.N * self.M),
                             mask=x.mask,
                             local_shapes=local_shapes,
                             partition=Partition.SCATTER,
                             dtype=self.dtype,
                             base_comm=self.base_comm
                             )

        # Calculate expected padded dimensions for x
        bk = self._K_padded // self._P_prime  # block size in K dimension

        # The input x corresponds to blocks from matrix B (K x M)
        # This process should receive a block of size (local_k x local_m)
        local_k = bk if self._row_id != self._P_prime - 1 else self.K - (self._P_prime - 1) * bk

        # Reshape x.local_array to its 2D block form
        x_block = x.local_array.reshape((local_k, local_m))

        # Pad the block to the full padded size if necessary
        pad_k = bk - local_k
        pad_m = bm - local_m

        if pad_k > 0 or pad_m > 0:
            x_block = np.pad(x_block, [(0, pad_k), (0, pad_m)], mode='constant')

        Y_local = np.zeros((self.A.shape[0], bm))

        for k in range(self._P_prime):
            Atemp = self.A.copy() if self._col_id == k else np.empty_like(self.A)
            Xtemp = x_block.copy() if self._row_id == k else np.empty_like(x_block)
            self._row_comm.bcast(Atemp, root=k)
            self._col_comm.bcast(Xtemp, root=k)
            Y_local += ncp.dot(Atemp, Xtemp)

        Y_local_unpadded = Y_local[:local_n, :local_m]
        y[:] = Y_local_unpadded.flatten()
        return y

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER}. Got {x.partition} instead.")

        # Calculate local shapes for block distribution
        bk = self._K_padded // self._P_prime  # block size in K dimension
        bm = self._M_padded // self._P_prime  # block size in M dimension

        # Calculate actual local shape for this process (considering original dimensions)
        # Adjust for edge/corner processes that might have smaller blocks
        local_k = bk if self._row_id != self._P_prime - 1 else self.K - (self._P_prime - 1) * bk
        local_m = bm if self._col_id != self._P_prime - 1 else self.M - (self._P_prime - 1) * bm

        local_shapes = self.base_comm.allgather(local_k * local_m)

        y = DistributedArray(
            global_shape=(self.K * self.M),
            mask=x.mask,
            local_shapes=local_shapes,
            partition=Partition.SCATTER,
            dtype=self.dtype,
            base_comm=self.base_comm
        )

        # Calculate expected padded dimensions for x
        bn = self._N_padded // self._P_prime  # block size in N dimension

        # The input x corresponds to blocks from the result (N x M)
        # This process should receive a block of size (local_n x local_m)
        local_n = bn if self._row_id != self._P_prime - 1 else self.N - (self._P_prime - 1) * bn

        # Reshape x.local_array to its 2D block form
        x_block = x.local_array.reshape((local_n, local_m))

        # Pad the block to the full padded size if necessary
        pad_n = bn - local_n
        pad_m = bm - local_m

        if pad_n > 0 or pad_m > 0:
            x_block = np.pad(x_block, [(0, pad_n), (0, pad_m)], mode='constant')

        A_local = self.At if hasattr(self, "At") else self.A.T.conj()
        Y_local = np.zeros((self.A.shape[1], bm))

        for k in range(self._P_prime):
            requests = []
            ATtemp = np.empty_like(A_local)
            srcA = k * self._P_prime + self._row_id
            tagA = (100 + k) * 1000 + self.rank
            requests.append(self.base_comm.Irecv(ATtemp, source=srcA, tag=tagA))
            if self._row_id == k:
                fixed_col = self._col_id
                for moving_col in range(self._P_prime):
                    destA = fixed_col * self._P_prime + moving_col
                    tagA = (100 + k) * 1000 + destA
                    requests.append(self.base_comm.Isend(A_local, dest=destA, tag=tagA))
            Xtemp = x_block.copy() if self._row_id == k else np.empty_like(x_block)
            requests.append(self._col_comm.Ibcast(Xtemp, root=k))
            MPI.Request.Waitall(requests)
            Y_local += ncp.dot(ATtemp, Xtemp)

        Y_local_unpadded = Y_local[:local_k, :local_m]
        y[:] = Y_local_unpadded.flatten()
        return y