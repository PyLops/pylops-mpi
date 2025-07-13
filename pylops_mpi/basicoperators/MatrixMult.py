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
        self._P_prime =  math.isqrt(size)
        if self._P_prime * self._P_prime != size:
            raise Exception(f"Number of processes must be a square number, provided {size} instead...")

        self._row_id, self._col_id =  divmod(rank, self._P_prime)

        self.base_comm = base_comm
        self._row_comm = base_comm.Split(color=self._row_id, key=self._col_id)
        self._col_comm = base_comm.Split(color=self._col_id, key=self._row_id)

        self.A = A.astype(np.dtype(dtype))

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

        if pr < 0 or pc < 0:
            raise Exception(f"Improper distribution of A expected local shape "
                            f"( ≤ {bn}, ≤ {bk}) but got ({A.shape[0]},{A.shape[1]})")

        if pr > 0 or pc > 0:
            self.A = np.pad(self.A, [(0, pr), (0, pc)], mode='constant')

        if saveAt:
            self.At = self.A.T.conj()

        self.dims  = (self.K, self.M)
        self.dimsd = (self.N, self.M)
        shape = (int(np.prod(self.dimsd)), int(np.prod(self.dims)))
        super().__init__(shape=shape, dtype=np.dtype(dtype), base_comm=base_comm)

    @staticmethod
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

    @staticmethod
    def block_distribute(array, proc_i, proc_j, comm):
        p_prime = math.isqrt(comm.Get_size())
        orig_r, orig_c = array.shape

        new_r = math.ceil(orig_r / p_prime) * p_prime
        new_c = math.ceil(orig_c / p_prime) * p_prime

        br, bc = new_r // p_prime, new_c // p_prime
        i0, j0 = proc_i * br, proc_j * bc
        i1, j1 = min(i0 + br, orig_r), min(j0 + bc, orig_c)

        i_end = None if proc_i == p_prime - 1 else i1
        j_end = None if proc_j == p_prime - 1 else j1
        block = array[i0:i_end, j0:j_end]

        pr = (new_r - orig_r) if proc_i == p_prime - 1 else 0
        pc = (new_c - orig_c) if proc_j == p_prime - 1 else 0
        #comment the padding to get the block as unpadded
        # if pr or pc: block = np.pad(block, [(0, pr), (0, pc)], mode='constant')
        return block, (new_r, new_c)

    @staticmethod
    def block_gather(x, new_shape, orig_shape, comm):
        ncp = get_module(x.engine)
        p_prime = math.isqrt(comm.Get_size())
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
            C[start_row:start_row + block_rows, start_col:start_col + block_cols] = block
        return C[:orr, :orc]

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER} Got {x.partition} instead...")

        # Calculate local shapes for block distribution
        bn = self._N_padded // self._P_prime  # block size in N dimension
        bm = self._M_padded // self._P_prime  # block size in M dimension

        # Calculate actual local shape for this process (considering original dimensions)
        local_n = bn
        local_m = bm

        # Adjust for edge/corner processes that might have smaller blocks
        if self._row_id == self._P_prime - 1:
            local_n = self.N - (self._P_prime - 1) * bn
        if self._col_id == self._P_prime - 1:
            local_m = self.M - (self._P_prime - 1) * bm

        local_shape = local_n * local_m

        # Create local_shapes array for all processes
        local_shapes = []
        for rank in range(self.size):
            row_id, col_id = divmod(rank, self._P_prime)
            proc_n = bn if row_id != self._P_prime - 1 else self.N - (self._P_prime - 1) * bn
            proc_m = bm if col_id != self._P_prime - 1 else self.M - (self._P_prime - 1) * bm
            local_shapes.append(proc_n * proc_m)

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
        local_k = bk
        if self._row_id == self._P_prime - 1:
            local_k = self.K - (self._P_prime - 1) * bk

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
            self._row_comm.Bcast(Atemp, root=k)
            self._col_comm.Bcast(Xtemp, root=k)
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
        local_k = bk
        local_m = bm

        # Adjust for edge/corner processes that might have smaller blocks
        if self._row_id == self._P_prime - 1:
            local_k = self.K - (self._P_prime - 1) * bk
        if self._col_id == self._P_prime - 1:
            local_m = self.M - (self._P_prime - 1) * bm

        local_shape = local_k * local_m

        # Create local_shapes array for all processes
        local_shapes = []
        for rank in range(self.size):
            row_id, col_id = divmod(rank, self._P_prime)
            proc_k = bk if row_id != self._P_prime - 1 else self.K - (self._P_prime - 1) * bk
            proc_m = bm if col_id != self._P_prime - 1 else self.M - (self._P_prime - 1) * bm
            local_shapes.append(proc_k * proc_m)

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
        local_n = bn
        if self._row_id == self._P_prime - 1:
            local_n = self.N - (self._P_prime - 1) * bn

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