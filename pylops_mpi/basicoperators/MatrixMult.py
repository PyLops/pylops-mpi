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
        where ``N_loc`` is the number of rows stored on this MPI rank and
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
        If the operator is created without a square number of mpi ranks.
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
    matrix and reshaped model and data vectors replicated across math:`P` 
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

    2. **Data Broadcasting**: Within each row (processes with same ``row_id``),
       the operand data is broadcast from the process whose ``col_id`` matches
       the ``row_id`` (processes along the diagonal). This ensures all processes
       in a row have access to the same operand columns.

    3. **Local Computation**: Each process computes ``A_local @ X_local`` where:
       - ``A_local`` is the local block of matrix ``A`` (shape ``N_local x K``)
       - ``X_local`` is the broadcasted operand (shape ``K x M_local``)

    4. **Row-wise Gather**: Results from all processes in each row are gathered
       using ``allgather`` to reconstruct the full result matrix vertically.

    **Adjoint Operation step-by-step**

    The adjoint operation performs the conjugate transpose multiplication:

    1. **Input Reshaping**: The input vector ``x`` is reshaped to ``(N, M_local)``
       representing the local columns of the input matrix.

    2. **Local Adjoint Computation**:
        Each process computes ``A_local.H @ X_tile``
            where ``A_local.H`` is either:
                - Pre-computed ``At`` (if ``saveAt=True``)
                - Computed on-the-fly as ``A.T.conj()`` (if ``saveAt=False``)
        Each process multiplies its transposed  local ``A`` block ``A_local^H`` 
        (shape ``K x N_block``)
        with the extracted  ``X_tile`` (shape ``N_block x M_local``),
        producing a partial result of  shape ``(K, M_local)``.
        This computes the local contribution of columns of  ``A^H`` to the final result.

    3. **Row-wise Reduction**: Since the full result ``Y = A^H \cdot X`` is the
       sum of contributions from all column blocks of ``A^H``, processes in the 
       same rows perform an ``allreduce`` sum to combine their partial results.
       This gives the complete ``(K, M_local)`` result for their assigned columns.
    
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

        block = array[i0:i1, j0:j1]
        pr = (new_r - orig_r) if proc_i == p_prime - 1 else 0
        pc = (new_c - orig_c) if proc_j == p_prime - 1 else 0
        if pr or pc:
            block = np.pad(block, [(0, pr), (0, pc)], mode='constant')

        return block, (new_r, new_c)

    @staticmethod
    def block_gather(x, new_shape, orig_shape, comm):
        ncp = get_module(x.engine)
        p_prime = math.isqrt(comm.Get_size())
        all_blks = comm.allgather(x.local_array)
        nr, nc   = new_shape
        orr, orc = orig_shape
        br, bc = nr // p_prime, nc // p_prime
        C = ncp.array(all_blks).reshape(p_prime, p_prime, br, bc).transpose(0, 2, 1, 3).reshape(nr, nc)
        return C[:orr, :orc]


    def _matvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER} Got {x.partition} instead...")
        local_shape = (self.N  // self._P_prime) * ( self.M * self._P_prime // self.size)
        y = DistributedArray(global_shape=((self.N // self._P_prime) * self.M * self._P_prime),
                             mask=x.mask,
                             local_shapes=[ local_shape for _ in range(self.size)],
                             partition=Partition.SCATTER,
                             dtype=self.dtype)

        x = x.local_array.reshape((self.A.shape[1], -1))
        c_local = np.zeros((self.A.shape[0], x.shape[1]))
        for k in range(self._P_prime):
            Atemp = self.A.copy() if self._col_id == k else np.empty_like(self.A)
            Xtemp = x.copy() if self._row_id == k else np.empty_like(x)
            self._row_comm.Bcast(Atemp, root=k)
            self._col_comm.Bcast(Xtemp, root=k)
            c_local += ncp.dot(Atemp, Xtemp)
        y[:] = c_local.flatten()
        return y


    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER}. Got {x.partition} instead.")

        local_shape = (self.K // self._P_prime) * (self.M * self._P_prime // self.size)
        y = DistributedArray(
            global_shape=((self.K // self._P_prime) * self.M * self._P_prime),
            mask=x.mask,
            local_shapes=[local_shape for _ in range(self.size)],
            partition=Partition.SCATTER,
            dtype=self.dtype,
        )
        x_reshaped = x.local_array.reshape((self.A.shape[0], -1))
        A_local = self.At if hasattr(self, "At") else self.A.T.conj()
        c_local = np.zeros((self.A.shape[1], x_reshaped.shape[1]))
        P = self._P_prime

        for k in range(P):
            temps = {}
            requests = []
            for buf, owner, base, name in (
                    (A_local, self._row_id, 100, 'A'),
                    (x_reshaped, self._col_id, 200, 'B'),
            ):
                tmp = np.empty_like(buf)
                temps[name] = tmp
                src, tag = k * P + owner, (base + k) * 1000 + self.rank
                requests.append(self.base_comm.Irecv(tmp, source=src, tag=tag))

                if self.rank // P == k:
                    fixed = self.rank % P
                    for moving in range(P):
                        dest = (fixed * P + moving) if name == 'A' else moving * P + fixed
                        tag = (base + k) * 1000 + dest
                        requests.append(self.base_comm.Isend(buf, dest=dest, tag=tag))
            MPI.Request.Waitall(requests)
            c_local += ncp.dot(temps['A'], temps['B'])
        y[:] = c_local.flatten()
        return y