import math

import numpy as np
from mpi4py import MPI
from pylops.utils.backend import get_module

from pylops_mpi import DistributedArray, MPILinearOperator, Partition
from pylops_mpi.Distributed import DistributedMixIn


def halo_block_split(global_shape: tuple, comm, grid_shape: tuple = None) -> tuple:
    r"""Split a global array over a Cartesian process grid.

    Compute the local slice owned by the calling rank when ``global_shape`` is
    distributed over ``grid_shape``. This helper follows the same Cartesian
    partitioning used internally by :class:`MPIHalo`.

    Parameters
    ----------
    global_shape : :obj:`tuple`
        Shape of the global array before flattening.
    comm : :obj:`mpi4py.MPI.Comm`
        MPI communicator containing the ranks in the process grid.
    grid_shape : :obj:`tuple`, optional
        Number of ranks along each array axis. When ``None``, all ranks are
        placed along the last axis.

    Returns
    -------
    local_slice : :obj:`tuple`
        Tuple of :class:`slice` objects selecting the local block owned by the
        calling rank.

    Raises
    ------
    ValueError
        If ``grid_shape`` does not contain exactly ``comm.Get_size()`` ranks.

    """
    ndim = len(global_shape)
    size = comm.Get_size()
    # default: put all ranks on the last axis
    if grid_shape is None:
        grid_shape = (1,) * (ndim - 1) + (size,)
    if math.prod(grid_shape) != size:
        raise ValueError(f"grid_shape {grid_shape} does not match comm size {size}")

    cart = comm.Create_cart(grid_shape, periods=[False] * ndim, reorder=True)
    coords = cart.Get_coords(cart.Get_rank())

    slices = []
    for gdim, procs_on_axis, coord in zip(global_shape, grid_shape, coords):
        block_size = math.ceil(gdim / procs_on_axis)
        start = coord * block_size
        end = min(start + block_size, gdim)
        if coord == procs_on_axis - 1:
            sl = slice(start, None)
        else:
            sl = slice(start, end)
        slices.append(sl)
    return tuple(slices)


class MPIHalo(DistributedMixIn, MPILinearOperator):
    r"""MPI Halo

    Apply haloing to a flattened :class:`pylops_mpi.DistributedArray`. The
    Halo operator is applied over a Cartesian process grid, where each rank
    owns one local block of the global N-dimensional array.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension.
    halo : :obj:`int` or :obj:`tuple`
        Number of halo samples to add around local blocks. A scalar value
        applies the same halo to both sides of every axis. A tuple of length
        ``ndim`` applies a symmetric halo per axis. A tuple of length
        ``2 * ndim`` specifies the halo before and after each axis.
    proc_grid_shape : :obj:`tuple`
        Number of MPI ranks along each dimension.
    comm : :obj:`mpi4py.MPI.Comm`, optional
        MPI Base Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
    dtype : :obj:`str`, optional
        Type of elements in the input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape

    Notes
    -----
    The MPIHalo operator supplies each rank with exactly the data dependencies that
    operators require when their output at a point depends only on a local neighborhood
    of radius :math:`h`. It augments the local array with a halo of width :math:`h`:
    for a rank :math:`p` that owns a set of grid points, the extended set consists
    of those owned points plus all points within a distance :math:`h` along the
    haloed axes, clipped to the global domain boundaries. In forward mode,halo samples
    are obtained by copying from the neighboring ranks that own them to rank :math:`p.

    The halo operator supplies the local data dependencies that some operators
    may require. For example, a finite-difference stencil or compact convolution
    requires only samples within a radius h to compute the output at a point
    :math:`i` owned by rank :math:`p`, then the complete local result on the owned
    points can be computed directly from the haloed block. The operator can
    therefore be applied independently on each rank after the exchange, commonly
    through :class:`pylops_mpi.basicoperators.MPIBlockDiag`.

    In one dimension, if rank :math:`p` owns the half-open interval :math:`[s_p, e_p)`
    and ``halo=1``, an interior rank receives :math:`x_{s_p-1}` from the previous
    rank and :math:`x_{e_p}` from the next rank. Boundary ranks omit samples outside
    the global domain. In multiple dimensions, the same extension is applied
    independently along each axis of the Cartesian process grid.

    The reverse operation is implemented by the adjoint: it extracts the original
    local domain by it removing the ghost cells and returning the local core.
    """

    def __init__(
        self,
        dims: tuple,
        halo,
        proc_grid_shape: tuple = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
        dtype=np.float64,
    ):
        self.global_dims = tuple(dims)
        self.ndim = len(dims)

        self.comm = comm
        self.dtype = dtype

        self.proc_grid_shape = tuple(proc_grid_shape)

        self.cart_comm, self.neigh = self._build_topo()
        self.halo = self._parse_halo(halo)

        self.local_dims = self._calc_local_dims()
        self.local_extent = self._calc_local_extent()
        self._local_extent_sizes = self._allgather(
            self.comm,
            None,
            int(np.prod(self.local_extent)),
        )

        self.shape = (
            int(np.sum(self._local_extent_sizes)),
            int(np.prod(self.global_dims)),
        )
        super().__init__(shape=self.shape, dtype=np.dtype(dtype), base_comm=comm)

    def _parse_halo(self, h):
        if isinstance(h, (int, np.int64, np.int32)):
            halo = (h,) * (2 * self.ndim)
            trimmed = list(halo)
            for ax in range(self.ndim):
                if trimmed[2 * ax] and self.neigh[("-", ax)] == MPI.PROC_NULL:
                    trimmed[2 * ax] = 0
                if trimmed[2 * ax + 1] and self.neigh[("+", ax)] == MPI.PROC_NULL:
                    trimmed[2 * ax + 1] = 0
            return tuple(trimmed)

        h = tuple(h)
        if len(h) == 1:
            halo = h * (2 * self.ndim)
        elif len(h) == self.ndim:
            halo = sum(tuple([(d, d) for d in h]), ())
        elif len(h) == 2 * self.ndim:
            halo = h
        else:
            raise ValueError(f"Invalid halo length {len(h)} for ndim={self.ndim}")
        return halo

    def _build_topo(self):
        cart_comm = self.comm.Create_cart(
            self.proc_grid_shape,
            periods=[False] * self.ndim,
            reorder=True,
        )
        neigh = {}
        for ax in range(self.ndim):
            before, after = cart_comm.Shift(ax, 1)
            neigh[("-", ax)] = before
            neigh[("+", ax)] = after
        return cart_comm, neigh

    def _calc_local_dims(self):
        rank = self.cart_comm.Get_rank()
        coords = self.cart_comm.Get_coords(rank)
        local = []
        for ax, (gdim, coord, grid_procs) in enumerate(
            zip(self.global_dims, coords, self.proc_grid_shape)
        ):
            block_size = math.ceil(gdim / grid_procs)
            start = coord * block_size
            end = min(start + block_size, gdim)
            local.append(end - start)
        return tuple(local)

    def _calc_local_extent(self):
        ext = []
        for ax in range(self.ndim):
            minus_halo, plus_halo = self.halo[2 * ax], self.halo[2 * ax + 1]
            ext.append(self.local_dims[ax] + minus_halo + plus_halo)
        return tuple(ext)

    def _exchange_along_axis(self, ncp, arr, axis, before, after):
        minus_nbr, plus_nbr = self.neigh[("-", axis)], self.neigh[("+", axis)]
        # slice definitions
        slicer = [slice(None)] * self.ndim
        # send before
        if before and minus_nbr != MPI.PROC_NULL:
            snd_s = slicer.copy()
            snd_s[axis] = slice(before, 2 * before)
            snd = arr[tuple(snd_s)].copy()
            rcv = ncp.empty_like(snd)
            self.cart_comm.Sendrecv(snd, dest=minus_nbr, recvbuf=rcv, source=minus_nbr)
            rcv_s = slicer.copy()
            rcv_s[axis] = slice(0, before)
            arr[tuple(rcv_s)] = rcv
        # send after
        if after and plus_nbr != MPI.PROC_NULL:
            snd_s = slicer.copy()
            snd_s[axis] = slice(-2 * after, -after)
            rcv_s = slicer.copy()
            rcv_s[axis] = slice(-after, None)
            snd = arr[tuple(snd_s)].copy()
            rcv = ncp.empty_like(snd)
            self.cart_comm.Sendrecv(snd, dest=plus_nbr, recvbuf=rcv, source=plus_nbr)
            arr[tuple(rcv_s)] = rcv

    def _matvec(self, x):
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(
                f"x should have partition={Partition.SCATTER} Got {x.partition} instead..."
            )

        y = DistributedArray(
            global_shape=self.shape[0],
            partition=Partition.SCATTER,
            local_shapes=self._local_extent_sizes,
            base_comm=x.base_comm,
            base_comm_nccl=x.base_comm_nccl,
            engine=x.engine,
            dtype=self.dtype,
        )

        core = x.local_array.reshape(self.local_dims)
        halo_arr = ncp.zeros(self.local_extent, dtype=self.dtype)
        # insert core
        core_slices = [
            slice(left, left + ldim)
            for left, ldim in zip(self.halo[::2], self.local_dims)
        ]
        halo_arr[tuple(core_slices)] = core

        # exchange along each axis
        for ax in range(self.ndim):
            before, after = self.halo[2 * ax], self.halo[2 * ax + 1]
            self._exchange_along_axis(
                ncp, halo_arr, axis=ax, before=before, after=after
            )

        y[:] = halo_arr.ravel()
        return y

    def _rmatvec(self, x):
        if x.partition != Partition.SCATTER:
            raise ValueError(
                f"x should have partition={Partition.SCATTER} Got {x.partition} instead..."
            )
        res = DistributedArray(
            global_shape=self.shape[1],
            partition=Partition.SCATTER,
            base_comm=x.base_comm,
            base_comm_nccl=x.base_comm_nccl,
            engine=x.engine,
            dtype=self.dtype,
        )
        arr = x.local_array.reshape(self.local_extent)
        core_slices = [
            slice(left, left + ldim)
            for left, ldim in zip(self.halo[::2], self.local_dims)
        ]
        core = arr[tuple(core_slices)]
        res[:] = core.ravel()
        return res
