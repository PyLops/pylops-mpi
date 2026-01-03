import math
import numpy as np

from mpi4py import MPI
from typing import Optional, Any, Tuple
from pylops.utils.backend import get_module
from pylops.utils.typing import DTypeLike, NDArray

from pylops_mpi import (
    DistributedArray,
    MPILinearOperator,
    Partition
)
from pylops_mpi.Distributed import DistributedMixIn
from pylops_mpi.DistributedArray import subcomm_split


def _choose_pb_and_p(P: int, nsl: int) -> Tuple[int, int]:
    """
    Choose Pb to minimize the α–β model under constraint P/Pb is a perfect square.
    Heuristic: largest Pb <= nsl such that P % Pb == 0 and is_square(P/Pb).
    """
    best = None
    for Pb in range(min(P, nsl), 0, -1):
        if P % Pb != 0: continue
        P2 = P // Pb
        p = int(math.isqrt(P2))
        if p * p == P2:
            best = (Pb, p)
            break
    if best is None:
        raise ValueError(
            f"No valid (Pb,p) with Pb<=nsl and P/Pb square. P={P}, nsl={nsl}."
        )
    return best


class MPIFredholm1SUMMA(DistributedMixIn, MPILinearOperator):
    """
    Distributed Fredholm-1 using batched SUMMA on contraction:
        d[k,:,:] = G[k,:,:] @ m[k,:,:]

    G is distributed as tiles (batch, x_tile, y_tile) over (batch_group, grid_row, grid_col).
    m is distributed as tiles (batch, y_tile, z_tile) over (batch_group, grid_row, grid_col).
    d is distributed as tiles (batch, x_tile, z_tile) over (batch_group, grid_row, grid_col).

    This operator uses Partition.SCATTER for both input and output.

    Parameters
    ----------
    G_local : ndarray
        Local tile of G of shape (B_g, nx_loc, ny_loc) for this rank.
    nz : int
        Global nz dimension.
    nsl_global : int, optional
        Global number of slices. If None, inferred from batch sizes across ranks.
    saveGt : bool, optional
        Save local conjugate-transpose of G tile for adjoint.
    pb : int, optional
        Number of batch groups. If None, auto-chosen.
    base_comm : MPI.Comm
    base_comm_nccl : optional NCCL comm (only if base_comm == COMM_WORLD)
    dtype : str
    """
    def __init__(
        self,
        G_local: NDArray,
        nz: int,
        nsl_global: Optional[int] = None,
        saveGt: bool = False,
        pb: Optional[int] = None,
        base_comm: MPI.Comm = MPI.COMM_WORLD,
        base_comm_nccl: Optional[Any] = None,
        dtype: DTypeLike = "float64",
    ) -> None:
        if base_comm_nccl is not None and base_comm is not MPI.COMM_WORLD:
            raise ValueError("base_comm_nccl requires base_comm=MPI.COMM_WORLD")

        self.base_comm = base_comm
        self.base_comm_nccl = base_comm_nccl
        self.rank = base_comm.Get_rank()
        self.size = base_comm.Get_size()

        # Local batch size
        if G_local.ndim != 3: raise ValueError(f"G_local must be 3D (B,nx_loc,ny_loc). Got {G_local.shape}")
        self.B  = int(G_local.shape[0])
        self.nz = int(nz)

        # Determine batch-grouping (Pb) and inner SUMMA grid size (p)
        # Need nsl_global for optimal choice; if not provided, use a conservative estimate from all ranks:
        if nsl_global is None:
            # Infer: sum(B_rank) over all ranks = p^2 * nsl_global (only true after we pick p)
            # So we first pick Pb,p using nsl_est = sum(B_rank)/min_square_factor_guess
            # Practical approach: assume Pb=1 initially, require P square, then compute nsl_global
            # If user doesn't provide nsl_global, we do *no* auto-optimization; pb must be given or P must be square
            if pb is None:
                p0 = int(math.isqrt(self.size))
                if p0 * p0 != self.size:
                    raise ValueError(
                        "If nsl_global is not provided, pb must be provided, "
                        "or P must be a perfect square (so pb=1 is valid)."
                    )
                pb = 1

        if pb is None:
            pb, p = _choose_pb_and_p(self.size, int(nsl_global))
        else:
            # For now we error but we could do something like where we would deactivate certain procs
            if self.size % pb != 0:
                raise ValueError(f"pb must divide P. Got pb={pb}, P={self.size}.")
            P2 = self.size // pb
            p = int(math.isqrt(P2))
            if p * p != P2:
                raise ValueError(f"P/pb must be a perfect square. Got P/pb={P2}.")
            if nsl_global is not None and pb > nsl_global:
                raise ValueError(f"pb must be <= nsl_global. Got pb={pb}, nsl_global={nsl_global}.")

        self.pb = int(pb)
        self.p = int(p)
        self.P2 = self.p * self.p

        # Batch-group id and rank within group
        self.batch_id = self.rank // self.P2
        self.rank_in_group = self.rank % self.P2

        if self.batch_id >= self.pb:
            raise ValueError(
                f"Rank mapping expects P == pb*p^2. "
                f"Got P={self.size}, pb={self.pb}, p={self.p} => pb*p^2={self.pb*self.P2}."
            )

        # Create batch communicator
        self.batch_comm = base_comm.Split(color=self.batch_id, key=self.rank_in_group)

        # Within group, 2D grid coords
        self.row_id, self.col_id = divmod(self.rank_in_group, self.p)

        # Row/col communicators (within group)
        self.row_comm = self.batch_comm.Split(color=self.row_id, key=self.col_id)
        self.col_comm = self.batch_comm.Split(color=self.col_id, key=self.row_id)

        # # NCCL subcomms if provided
        # if base_comm_nccl is not None:
        #     # subcomm_split expects mask per WORLD rank
        #     # batch_comm: group by batch_id
        #     mask_batch = [r // self.P2 for r in range(self.size)]
        #     self.batch_comm_nccl = subcomm_split(mask_batch, base_comm_nccl)
        #
        #     # row_comm: group by (batch_id,row_id)
        #     mask_row = []
        #     mask_col = []
        #     for r in range(self.size):
        #         bid = r // self.P2
        #         rig = r % self.P2
        #         rr, cc = divmod(rig, self.p)
        #         mask_row.append(bid * self.p + rr)
        #         mask_col.append(bid * self.p + cc)
        #     self.row_comm_nccl = subcomm_split(mask_row, base_comm_nccl)
        #     self.col_comm_nccl = subcomm_split(mask_col, base_comm_nccl)
        # else:
        self.batch_comm_nccl = None
        self.row_comm_nccl = None
        self.col_comm_nccl = None

        # Store G tile and optional GT
        self.G = G_local.astype(np.dtype(dtype))
        if saveGt:
            # (B, nx_loc, ny_loc) -> (B, ny_loc, nx_loc)
            self.GT = self.G.transpose(0, 2, 1).conj()

        # Infer global nx, ny from within-group tiling
        # A tile: (nx_loc, ny_loc) where nx is reduced on col_comm, ny on row_comm
        nx_loc = self.G.shape[1]
        ny_loc = self.G.shape[2]
        self.nx = int(self.col_comm.allreduce(nx_loc, op=MPI.SUM))
        self.ny = int(self.row_comm.allreduce(ny_loc, op=MPI.SUM))

        # Determine global nsl
        if nsl_global is None:
            # sum B over WORLD ranks = p^2 * sum(B over batch groups) = p^2 * nsl_global
            Bsum = int(self.base_comm.allreduce(self.B, op=MPI.SUM))
            if Bsum % self.P2 != 0:
                raise ValueError(
                    f"Cannot infer nsl_global cleanly: sum(B)={Bsum} not divisible by p^2={self.P2}."
                )
            self.nsl = Bsum // self.P2
        else:
            self.nsl = int(nsl_global)

        # Padding sizes for SUMMA blocks
        self.nx_pad = math.ceil(self.nx / self.p) * self.p
        self.ny_pad = math.ceil(self.ny / self.p) * self.p
        self.nz_pad = math.ceil(self.nz / self.p) * self.p

        self.bn = self.nx_pad // self.p
        self.bk = self.ny_pad // self.p
        self.bm = self.nz_pad // self.p

        # Local (unpadded) extents for this rank’s output tile (x,z) and input tile (y,z)
        self.local_n = max(0, min(self.bn, self.nx - self.row_id * self.bn))
        self.local_k = max(0, min(self.bk, self.ny - self.row_id * self.bk))  # for m (K rows) uses row_id
        self.local_ka = max(0, min(self.bk, self.ny - self.col_id * self.bk))  # for G (K cols) uses col_id
        self.local_m = max(0, min(self.bm, self.nz - self.col_id * self.bm))

        # Operator global shapes (conceptual / unpadded)
        self.dims_model = (self.nsl, self.ny, self.nz)
        self.dims_data = (self.nsl, self.nx, self.nz)
        shape = (int(np.prod(self.dims_data)), int(np.prod(self.dims_model)))
        super().__init__(shape=shape, dtype=np.dtype(dtype), base_comm=base_comm)

        # Ensure local G matches expected tile sizes in (nx_loc, ny_loc) for A distribution
        # We allow edge tiles to be smaller since we will pad later
        if self.G.shape[1] != self.local_n or self.G.shape[2] != self.local_ka:
            # Not necessarily fatal if user pre-padded; allow larger, but disallow mismatch that breaks slicing
            if self.G.shape[1] < self.local_n or self.G.shape[2] < self.local_ka:
                raise ValueError(
                    f"G_local tile too small for this rank. "
                    f"Expected at least ({self.B},{self.local_n},{self.local_ka}), got {self.G.shape}."
                )

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER}. Got {x.partition} instead.")

        # Input local tile expected shape: (B, local_k (by row_id), local_m (by col_id))
        expected_in = self.B * self.local_k * self.local_m
        if x.local_array.size != expected_in:
            raise ValueError(
                f"Local x size mismatch. Expected {expected_in} elements "
                f"(B={self.B}, local_k={self.local_k}, local_m={self.local_m}), "
                f"got {x.local_array.size}."
            )

        output_dtype = np.result_type(self.dtype, x.dtype)

        # Output local shapes for SCATTER vector
        my_out = self.B * self.local_n * self.local_m
        local_shapes = self.base_comm.allgather(my_out)

        y = DistributedArray(
            global_shape=int(np.prod(self.dims_data)),
            local_shapes=local_shapes,
            mask=x.mask,
            partition=Partition.SCATTER,
            engine=x.engine,
            dtype=output_dtype,
            base_comm=x.base_comm,
            base_comm_nccl=x.base_comm_nccl,
        )

        # Reshape local x tile and pad to (B, bk, bm)
        X = x.local_array.reshape((self.B, self.local_k, self.local_m)).astype(output_dtype)
        if self.local_k != self.bk or self.local_m != self.bm:
            Xp = ncp.zeros((self.B, self.bk, self.bm), dtype=output_dtype)
            Xp[:, :self.local_k, :self.local_m] = X
            X = Xp

        # Pad local G tile to (B, bn, bk) for SUMMA A tiles
        G = self.G[:, :self.local_n, :self.local_ka].astype(output_dtype)
        if self.local_n != self.bn or self.local_ka != self.bk:
            Gp = ncp.zeros((self.B, self.bn, self.bk), dtype=output_dtype)
            Gp[:, :self.local_n, :self.local_ka] = G
            G = Gp

        Y = ncp.zeros((self.B, self.bn, self.bm), dtype=output_dtype)

        row_nccl = self.row_comm_nccl if x.engine == "cupy" else None
        col_nccl = self.col_comm_nccl if x.engine == "cupy" else None

        # Batched SUMMA
        for k in range(self.p):
            Atemp = G.copy() if self.col_id == k else ncp.empty_like(G)
            Btemp = X.copy() if self.row_id == k else ncp.empty_like(X)

            Atemp = self._bcast(self.row_comm, row_nccl, Atemp, root=k, engine=x.engine)
            Btemp = self._bcast(self.col_comm, col_nccl, Btemp, root=k, engine=x.engine)

            Y += ncp.matmul(Atemp, Btemp)

        # Unpad to local (B, local_n, local_m) and write out
        Y = Y[:, :self.local_n, :self.local_m]
        y[:] = Y.ravel()
        return y

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER}. Got {x.partition} instead.")

        # Input to adjoint is data tile: (B, local_n, local_m)
        expected_in = self.B * self.local_n * self.local_m
        if x.local_array.size != expected_in:
            raise ValueError(
                f"Local x size mismatch for adjoint. Expected {expected_in} elements "
                f"(B={self.B}, local_n={self.local_n}, local_m={self.local_m}), got {x.local_array.size}."
            )

        # Output dtype rules similar to your matrix-mult operators
        if np.iscomplexobj(self.G):
            output_dtype = np.result_type(self.dtype, x.dtype)
        else:
            output_dtype = x.dtype if np.iscomplexobj(x.local_array) else self.dtype
            output_dtype = np.result_type(self.dtype, output_dtype)

        # Output local shapes for SCATTER model vector
        my_out = self.B * self.local_k * self.local_m  # (B, local_k(row_id), local_m(col_id))
        local_shapes = self.base_comm.allgather(my_out)

        y = DistributedArray(
            global_shape=int(np.prod(self.dims_model)),
            local_shapes=local_shapes,
            mask=x.mask,
            partition=Partition.SCATTER,
            engine=x.engine,
            dtype=output_dtype,
            base_comm=x.base_comm,
            base_comm_nccl=x.base_comm_nccl,
        )

        # Reshape x tile and pad to (B, bn, bm)
        X = x.local_array.reshape((self.B, self.local_n, self.local_m)).astype(output_dtype)
        if self.local_n != self.bn or self.local_m != self.bm:
            Xp = ncp.zeros((self.B, self.bn, self.bm), dtype=output_dtype)
            Xp[:, :self.local_n, :self.local_m] = X
            X = Xp

        # Local A^H tile (transpose-conj of A tile): (B, bk, bn)
        if hasattr(self, "GT"):
            AT_local = self.GT[:, :self.local_ka, :self.local_n].astype(output_dtype)
        else:
            AT_local = self.G[:, :self.local_n, :self.local_ka].transpose(0, 2, 1).conj().astype(output_dtype)

        if self.local_ka != self.bk or self.local_n != self.bn:
            ATp = ncp.zeros((self.B, self.bk, self.bn), dtype=output_dtype)
            ATp[:, :self.local_ka, :self.local_n] = AT_local
            AT_local = ATp
        AT_local = ncp.ascontiguousarray(AT_local)

        Y = ncp.zeros((self.B, self.bk, self.bm), dtype=output_dtype)

        base_nccl = self.base_comm_nccl if x.engine == "cupy" else None
        col_nccl = self.col_comm_nccl if x.engine == "cupy" else None

        # Batched adjoint SUMMA variant matching your existing _MPISummaMatrixMult._rmatvec:
        # - broadcast X panels down col_comm
        # - move AT blocks across WORLD ranks to emulate transposed distribution
        for k in range(self.p):
            Xtemp = X.copy() if self.row_id == k else ncp.empty_like(X)
            Xtemp = self._bcast(self.col_comm, col_nccl, Xtemp, root=k, engine=x.engine)

            # Determine source rank for AT block needed this iteration
            # WORLD rank mapping inside batch group:
            #   world_rank = batch_id*P2 + (row*p + col)
            # Need AT from srcA = (row=k, col=row_id) within this batch group:
            srcA_in_group = k * self.p + self.row_id
            srcA = self.batch_id * self.P2 + srcA_in_group

            ATtemp = AT_local if (self.rank == srcA) else None

            # Send from ranks with row_id==k (within group) to row=col_id targets, across all columns (within group),
            # using WORLD communicator for explicit point-to-point
            for moving_col in range(self.p):
                if self.row_id == k:
                    # sender is (row=k, col=self.col_id)
                    dest_in_group = self.col_id * self.p + moving_col
                    destA = self.batch_id * self.P2 + dest_in_group
                    if destA != self.rank:
                        tagA = (100 + k) * 100000 + destA
                        self._send(self.base_comm, base_nccl, AT_local, dest=destA, tag=tagA, engine=x.engine)

                if self.col_id == moving_col and ATtemp is None:
                    tagA = (100 + k) * 100000 + self.rank
                    recv_buf = ncp.empty_like(AT_local)
                    ATtemp = self._recv(self.base_comm, base_nccl, recv_buf, source=srcA, tag=tagA, engine=x.engine)

            Y += ncp.matmul(ATtemp, Xtemp)

        # Unpad output to (B, local_k(row_id), local_m)
        Y = Y[:, :self.local_k, :self.local_m]
        y[:] = Y.ravel()
        return y

class MPIFredholm1(MPILinearOperator):
    r"""Fredholm integral of first kind.

    Implement a multi-dimensional Fredholm integral of first kind distributed
    across the first dimension

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel of size
        :math:`[n_{\text{slice}} \times n_x \times n_y]`
    nz : :obj:`int`, optional
        Additional dimension of model
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G.H`` to speed up the computation of adjoint
        (``True``) or create ``G.H`` on-the-fly (``False``)
        Note that ``saveGt=True`` will double the amount of required memory
    usematmul : :obj:`bool`, optional
        Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
        (``False``). As it is not possible to define which approach is more
        performant (this is highly dependent on the size of ``G`` and input
        arrays as well as the hardware used in the computation), we advise users
        to time both methods for their specific problem prior to making a
        choice.
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
    NotImplementedError
        If the size of the first dimension of ``G`` is equal to 1 in any of the ranks

    Notes
    -----
    A multi-dimensional Fredholm integral of first kind can be expressed as

    .. math::

        d(k, x, z) = \int{G(k, x, y) m(k, y, z) \,\mathrm{d}y}
        \quad \forall k=1,\ldots,n_{slice}

    on the other hand its adjoint is expressed as

    .. math::

        m(k, y, z) = \int{G^*(k, y, x) d(k, x, z) \,\mathrm{d}x}
        \quad \forall k=1,\ldots,n_{\text{slice}}

    This integral is implemented in a distributed fashion, where ``G``
    is split across ranks along its first dimension. The inputs
    of both the forward and adjoint are distributed arrays with broadcast partion:
    each rank takes a portion of such arrays, computes a partial integral, and
    the resulting outputs are then gathered by all ranks to return a
    distributed arrays with broadcast partion.

    """

    def __init__(
        self,
        G: NDArray,
        nz: int = 1,
        saveGt: bool = False,
        usematmul: bool = True,
        base_comm: MPI.Comm = MPI.COMM_WORLD,
        dtype: DTypeLike = "float64",
    ) -> None:
        self.nz = nz
        self.nsl, self.nx, self.ny = G.shape
        self.nsls = base_comm.allgather(self.nsl)
        if base_comm.Get_rank() == 0 and 1 in self.nsls:
            raise NotImplementedError(f'All ranks must have at least 2 or more '
                                      f'elements in the first dimension: '
                                      f'local split is instead {self.nsls}...')
        nslstot = base_comm.allreduce(self.nsl)
        self.islstart = np.insert(np.cumsum(self.nsls)[:-1], 0, 0)
        self.islend = np.cumsum(self.nsls)
        self.rank = base_comm.Get_rank()
        self.dims = (nslstot, self.ny, self.nz)
        self.dimsd = (nslstot, self.nx, self.nz)
        shape = (np.prod(self.dimsd),
                 np.prod(self.dims))
        super().__init__(shape=shape, dtype=np.dtype(dtype), base_comm=base_comm)

        self.G = G
        if saveGt:
            self.GT = G.transpose((0, 2, 1)).conj()
        self.usematmul = usematmul

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition not in [Partition.BROADCAST, Partition.UNSAFE_BROADCAST]:
            raise ValueError(f"x should have partition={Partition.BROADCAST},{Partition.UNSAFE_BROADCAST}"
                             f"Got  {x.partition} instead...")
        y = DistributedArray(global_shape=self.shape[0],
                             base_comm=x.base_comm,
                             base_comm_nccl=x.base_comm_nccl,
                             partition=x.partition,
                             engine=x.engine, dtype=self.dtype)
        x = x.local_array.reshape(self.dims).squeeze()
        x = x[self.islstart[self.rank]:self.islend[self.rank]]
        # apply matmul for portion of the rank of interest
        if self.usematmul:
            if self.nz == 1:
                x = x[..., ncp.newaxis]
            y1 = ncp.matmul(self.G, x)
        else:
            y1 = ncp.squeeze(ncp.zeros((self.nsls[self.rank], self.nx, self.nz), dtype=self.dtype))
            for isl in range(self.nsls[self.rank]):
                y1[isl] = ncp.dot(self.G[isl], x[isl])
        # gather results
        y[:] = ncp.vstack(y._allgather(y.base_comm, y.base_comm_nccl, y1,
                                       engine=y.engine)).ravel()
        return y

    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_module(x.engine)
        if x.partition not in [Partition.BROADCAST, Partition.UNSAFE_BROADCAST]:
            raise ValueError(f"x should have partition={Partition.BROADCAST},{Partition.UNSAFE_BROADCAST}"
                             f"Got  {x.partition} instead...")
        y = DistributedArray(global_shape=self.shape[1],
                             base_comm=x.base_comm,
                             base_comm_nccl=x.base_comm_nccl,
                             partition=x.partition,
                             engine=x.engine, dtype=self.dtype)
        x = x.local_array.reshape(self.dimsd).squeeze()
        x = x[self.islstart[self.rank]:self.islend[self.rank]]
        # apply matmul for portion of the rank of interest
        if self.usematmul:
            if self.nz == 1:
                x = x[..., ncp.newaxis]
            if hasattr(self, "GT"):
                y1 = ncp.matmul(self.GT, x)
            else:
                y1 = (
                    ncp.matmul(x.transpose(0, 2, 1).conj(), self.G)
                    .transpose(0, 2, 1)
                    .conj()
                )
        else:
            y1 = ncp.squeeze(ncp.zeros((self.nsls[self.rank], self.ny, self.nz), dtype=self.dtype))
            if hasattr(self, "GT"):
                for isl in range(self.nsls[self.rank]):
                    y1[isl] = ncp.dot(self.GT[isl], x[isl])
            else:
                for isl in range(self.nsl):
                    y1[isl] = ncp.dot(x[isl].T.conj(), self.G[isl]).T.conj()

        # gather results
        y[:] = ncp.vstack(y._allgather(y.base_comm, y.base_comm_nccl, y1, engine=y.engine)).ravel()
        return y
