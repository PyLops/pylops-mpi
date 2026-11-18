import math
from typing import Any, Optional, Tuple

import numpy as np
from mpi4py import MPI
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
    """Choose the largest batch count that leaves a square grid per batch."""
    for pb in range(min(P, nsl), 0, -1):
        p = math.isqrt(P // pb)
        if pb * p * p == P:
            return pb, p
    raise ValueError(f"No square process grid for {P} ranks and {nsl} slices.")


class MPIFredholm1SUMMA(DistributedMixIn, MPILinearOperator):
    """Fredholm integral of the first kind using batched SUMMA.

    Parameters
    ----------
    G_local : numpy.ndarray or cupy.ndarray
        Unpadded kernel tile of shape (local slices, local x, local y).
    nz : int
        Global number of columns in each model and data slice.
    nsl_global : int, optional
        Total number of slices. By default, infer it from the batch groups.
    saveGt : bool, optional
        Store the conjugate transpose of the local kernel for adjoint products.
    pb : int, optional
        Number of batch groups. Each group must contain a square number of ranks.
        With a known slice count, the default is the largest valid group count
        up to that count. With an inferred slice count, the default is one group.
    base_comm : mpi4py.MPI.Comm, optional
        MPI communicator containing the batch groups.
    base_comm_nccl : cupy.cuda.nccl.NcclCommunicator, optional
        NCCL communicator for GPU transfers when using MPI.COMM_WORLD.
    dtype : str or numpy.dtype, optional
        Kernel dtype. Products use the common dtype of the kernel and vector.

    Notes
    -----
    Each slice evaluates ``data[s] = G[s] @ model[s]``. Consecutive groups of
    ranks own consecutive batches of slices. Within a group of ``p * p`` ranks,
    grid coordinates follow row-major order. Each matrix axis of length ``n``
    uses blocks of width ``ceil(n / p)``, clipped at the boundary.

    Supply model tiles over (slices, y, z) and data tiles over (slices, x, z).
    Each rank stores its tile flattened in C order in a one-dimensional
    :class:`pylops_mpi.DistributedArray` with ``Partition.SCATTER``.
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

        self.nz = int(nz)
        if self.nz < 0:
            raise ValueError("nz must be nonnegative.")
        if pb is None and nsl_global is not None:
            self.pb, self.p = _choose_pb_and_p(self.size, int(nsl_global))
        else:
            self.pb = 1 if pb is None else pb
            if self.pb < 1 or self.size % self.pb:
                raise ValueError(f"pb must be a positive divisor of {self.size}.")
            self.p = math.isqrt(self.size // self.pb)
            if self.pb * self.p ** 2 != self.size:
                raise ValueError("P/pb must be a perfect square.")
            if nsl_global is not None and self.pb > nsl_global:
                raise ValueError("pb must be <= nsl_global.")

        self.P2 = self.p ** 2
        self.batch_id, self.rank_in_group = divmod(self.rank, self.P2)

        self.row_id, self.col_id = divmod(self.rank_in_group, self.p)

        # Inspect all tile shapes before communicating panels. A bad tile must
        # raise on every rank, including ranks in other batch groups.
        shapes = base_comm.allgather(G_local.shape)
        if any(len(shape) != 3 for shape in shapes):
            raise ValueError("G_local must be a 3D (slices, x, y) tile.")
        shapes = np.asarray(shapes).reshape(self.pb, self.p, self.p, 3)
        batch_sizes = shapes[:, 0, 0, 0]
        self.B = int(batch_sizes[self.batch_id])
        self.nsl = int(batch_sizes.sum())
        if nsl_global is not None and self.nsl != nsl_global:
            raise ValueError("Batch sizes do not sum to nsl_global.")
        self.nx = int(shapes[0, :, 0, 1].sum())
        self.ny = int(shapes[0, 0, :, 2].sum())

        # Pad each matrix dimension to a multiple of the grid width.
        self.bn = (self.nx + self.p - 1) // self.p
        self.bk = (self.ny + self.p - 1) // self.p
        self.bm = (self.nz + self.p - 1) // self.p
        self.nx_pad, self.ny_pad, self.nz_pad = (
            self.bn * self.p, self.bk * self.p, self.bm * self.p
        )
        coords = np.arange(self.p)
        x_sizes = np.clip(self.nx - coords * self.bn, 0, self.bn)
        y_sizes = np.clip(self.ny - coords * self.bk, 0, self.bk)
        z_sizes = np.clip(self.nz - coords * self.bm, 0, self.bm)
        if not (np.all(shapes[..., 0] == batch_sizes[:, None, None])
                and np.all(shapes[..., 1] == x_sizes[None, :, None])
                and np.all(shapes[..., 2] == y_sizes[None, None, :])):
            raise ValueError("G_local shapes do not match the batch and grid layout.")
        self.local_n = x_sizes[self.row_id]
        self.local_k = y_sizes[self.row_id]
        self.local_ka = y_sizes[self.col_id]
        self.local_m = z_sizes[self.col_id]
        batch_cols = batch_sizes[:, None, None] * z_sizes[None, None, :]
        self._model_sizes = (batch_cols * y_sizes[None, :, None]).ravel().tolist()
        self._data_sizes = (batch_cols * x_sizes[None, :, None]).ravel().tolist()

        self.batch_comm = base_comm.Split(color=self.batch_id, key=self.rank_in_group)
        self.row_comm = self.batch_comm.Split(color=self.row_id, key=self.col_id)
        self.col_comm = self.batch_comm.Split(color=self.col_id, key=self.row_id)
        self.batch_comm_nccl = self.row_comm_nccl = self.col_comm_nccl = None
        if base_comm_nccl is not None:
            # Subcommunicator ranks follow the same ordering as their MPI peers.
            ranks = np.arange(self.size)
            groups = ranks // self.P2
            rows, cols = np.divmod(ranks % self.P2, self.p)
            self.batch_comm_nccl = subcomm_split(groups.tolist(), base_comm_nccl)
            self.row_comm_nccl = subcomm_split((groups * self.p + rows).tolist(), base_comm_nccl)
            self.col_comm_nccl = subcomm_split((groups * self.p + cols).tolist(), base_comm_nccl)

        self.G = G_local.astype(dtype)
        if saveGt:
            self.GT = self.G.transpose(0, 2, 1).conj()
        self.dims_model = (self.nsl, self.ny, self.nz)
        self.dims_data = (self.nsl, self.nx, self.nz)
        super().__init__(dims=self.dims_model, dimsd=self.dims_data,
                         dtype=np.dtype(dtype), base_comm=base_comm)

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        return self._multiply(x, adjoint=False)

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        return self._multiply(x, adjoint=True)

    def _multiply(self, x: DistributedArray, adjoint: bool) -> DistributedArray:
        ncp = get_module(x.engine)
        dtype = np.result_type(self.dtype, x.dtype)
        if adjoint:
            in_rows, out_rows = self.local_n, self.local_k
            in_block, out_block = self.bn, self.bk
            out_sizes, out_size = self._model_sizes, self.shape[1]
        else:
            in_rows, out_rows = self.local_k, self.local_n
            in_block, out_block = self.bk, self.bn
            out_sizes, out_size = self._data_sizes, self.shape[0]

        if MPI.Comm.Compare(self.base_comm, x.base_comm) not in (MPI.IDENT, MPI.CONGRUENT):
            raise ValueError("The vector must use the same MPI rank ordering as the operator.")
        if x.partition != Partition.SCATTER:
            raise ValueError("SUMMA requires a SCATTER vector.")
        valid = x.local_array.size == self.B * in_rows * self.local_m
        if not self.base_comm.allreduce(valid, op=MPI.LAND):
            raise ValueError("Local vector sizes do not match the SUMMA tiles.")

        def pad(tile, rows, cols):
            if tile.shape[1:] == (rows, cols):
                return ncp.ascontiguousarray(tile, dtype=dtype)
            padded = ncp.zeros((self.B, rows, cols), dtype=dtype)
            padded[:, :tile.shape[1], :tile.shape[2]] = tile
            return padded

        X = pad(x.local_array.reshape(self.B, in_rows, self.local_m), in_block, self.bm)
        if adjoint:
            GT = self.GT if hasattr(self, "GT") else self.G.transpose(0, 2, 1).conj()
            G = pad(GT, out_block, in_block)
            # Rank (i,j) needs the conjugate-transposed tile from (j,i).
            if self.row_id != self.col_id:
                peer = self.col_id * self.p + self.row_id
                comm_nccl = self.batch_comm_nccl if x.engine == "cupy" else None
                G = self._sendrecv(self.batch_comm, comm_nccl, G, ncp.empty_like(G),
                                   dest=peer, source=peer, engine=x.engine)
        else:
            G = pad(self.G, out_block, in_block)

        Y = ncp.zeros((self.B, out_block, self.bm), dtype=dtype)
        row_nccl = self.row_comm_nccl if x.engine == "cupy" else None
        col_nccl = self.col_comm_nccl if x.engine == "cupy" else None
        for k in range(self.p):
            A = G if self.col_id == k else ncp.empty_like(G)
            B = X if self.row_id == k else ncp.empty_like(X)
            A = self._bcast(self.row_comm, row_nccl, A, root=k, engine=x.engine)
            B = self._bcast(self.col_comm, col_nccl, B, root=k, engine=x.engine)
            Y += ncp.matmul(A, B)

        y = DistributedArray(
            global_shape=out_size, local_shapes=out_sizes, mask=x.mask,
            partition=Partition.SCATTER, engine=x.engine, dtype=dtype,
            base_comm=x.base_comm, base_comm_nccl=x.base_comm_nccl,
        )
        y[:] = Y[:, :out_rows, :self.local_m].ravel()
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
        dims = (nslstot, self.ny, self.nz)
        dimsd = (nslstot, self.nx, self.nz)
        super().__init__(dims=dims, dimsd=dimsd, dtype=np.dtype(dtype), base_comm=base_comm)

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

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
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
