import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from mpi4py import MPI
from pylops.utils.backend import get_module

from pylops_mpi import DistributedArray, MPILinearOperator, Partition
from pylops_mpi.Distributed import DistributedMixIn


def halo_block_split(
    global_shape: tuple,
    comm: MPI.Comm,
    grid_shape: Optional[tuple] = None,
) -> tuple:
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

    Apply haloing to all dimensions of a flattened, 1-dimensional
    :class:`pylops_mpi.DistributedArray` after local reshaping to a
    N-dimensional array.

    The Halo operator is applied over a Cartesian process grid, where each
    rank owns a local block of the global N-dimensional array.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension.
    halo : :obj:`int` or :obj:`tuple`
        Number of halo samples to add around each local block. A scalar value
        applies the same halo to both sides of every axis. A tuple of length
        ``ndim`` applies a symmetric halo per axis. A tuple of length
        ``2 * ndim`` specifies the halo to apply at the start and at the end
        for each axis.
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
    The MPIHalo operator extends each rank's local array with a **halo** (ghost cells)
    of width ``halo`` along the haloed axes, providing the neighboring rank's data for
    stencil-like operators. Ranks are arranged in an N-dimensional Cartesian grid as
    provided by ``proc_grid_shape``, whereby each rank owns a contiguous block of the
    global array. The ``halo`` is normalised to a tuple of length ``2 * ndim``, containing
    one ``(minus, plus)`` halo-width pair for each axis. The tuple is flattened in
    axis order as

    .. math::
        (h_{0,-}, h_{0,+}, h_{1,-}, h_{1,+}, \ldots)

    where :math:`h_{i,-}`` and :math:`h_{i,+}`` represent the halo widths on the
    negative and positive side of the i-th axis, respectively. For convenience,
    ``halo`` may be provided as a scalar when the same symmetric halo is
    required along all axes, or as a tuple of length ``ndim`` when symmetric halos
    of different width are applied to different axis. Ghost cells on the global
    boundary of an axis are zero by default.

    In the forward mode, each rank exchanges boundary slices with its left and
    right neighbors along each axis via ``MPI_Sendrecv``. Ranks at a global
    boundary have ``MPI.PROC_NULL`` as their neighbor on that side, so those
    ghost regions remain zero and no exchange is attempted. Once the exchange
    is complete, local PyLops operators can be applied independently
    on each rank's extended block, typically wrapped into a
    :class:`pylops_mpi.basicoperators.MPIBlockDiag` operator.

    In the adjoint mode, the reverse operation is performed the original
    local domain is extracted by removing the ghost cells.

    Finally, note that the Halo operator is not linear operator per se; instead,
    it is meant to sandwitch any linear operator to implement equivalent
    behaviours to the serial version of such an operator.

    """

    def __init__(
        self,
        dims: tuple,
        halo: Union[int, tuple],
        proc_grid_shape: Optional[tuple] = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
        dtype: Any = np.float64,
    ) -> None:
        self.global_dims = tuple(dims)
        self.ndim = len(dims)

        self.comm = comm
        self.dtype = dtype

        if proc_grid_shape is None:
            proc_grid_shape = (1,) * (self.ndim - 1) + (self.comm.Get_size(),)
        self.proc_grid_shape = tuple(proc_grid_shape)
        if math.prod(self.proc_grid_shape) != self.comm.Get_size():
            raise ValueError(
                f"grid_shape {self.proc_grid_shape} does not match comm size {self.comm.Get_size()}"
            )

        self.cart_comm, self.neigh = self._build_topo()
        self.halo = self._parse_halo(halo)

        self.local_dims = self._calc_local_dims()
        self.local_extent = self._calc_local_extent()
        self._validate_exchange_widths()
        self._local_dim_sizes = []
        # For uneven global dimensions, MPIHalo's Cartesian block sizes differ
        # from DistributedArray's default flat split. Store those sizes so
        # _rmatvec can build an adjoint output with the same local ownership
        # as the original Halo input.
        comm_group = self.comm.Get_group()
        cart_group = self.cart_comm.Get_group()
        for rank in range(self.comm.Get_size()):
            cart_rank = MPI.Group.Translate_ranks(comm_group, [rank], cart_group)[0]
            coords = self.cart_comm.Get_coords(cart_rank)
            local_size = 1
            for gdim, coord, grid_procs in zip(self.global_dims, coords, self.proc_grid_shape):
                block_size = math.ceil(gdim / grid_procs)
                start = coord * block_size
                end = min(start + block_size, gdim)
                local_size *= end - start
            self._local_dim_sizes.append(local_size)

        self._local_extent_sizes = self._allgather(
            self.comm,
            None,
            int(np.prod(self.local_extent)),
        )
        dims = self.global_dims
        dimsd = (self._allreduce(
            comm,
            None,
            np.prod(np.array(self.local_extent))
        ), )
        super().__init__(dims=dims, dimsd=dimsd, dtype=np.dtype(dtype), base_comm=comm)

    def _parse_halo(self, h: Union[int, tuple]) -> tuple:
        """Normalize halo input to a 2 * ndim tuple of per-side widths for each axis of the N-dimensional array.

        Accepts a scalar, a tuple of length-1, one value per axis (the same value is assigned to both sides),
        or explicit minus/plus pairs for each axis.
        """
        if isinstance(h, (int, np.int64, np.int32)):
            halo = (h,) * (2 * self.ndim)
            trimmed = list(halo)
            for ax in range(self.ndim):
                if trimmed[2 * ax] and self.neigh[("-", ax)] == MPI.PROC_NULL:
                    trimmed[2 * ax] = 0
                if trimmed[2 * ax + 1] and self.neigh[("+", ax)] == MPI.PROC_NULL:
                    trimmed[2 * ax + 1] = 0
            halo = tuple(trimmed)
            if any(h < 0 for h in halo):
                raise ValueError("Halo widths must be non-negative")
            return halo

        h = tuple(h)
        if len(h) == 1:
            halo = h * (2 * self.ndim)
        elif len(h) == self.ndim:
            halo = sum(tuple([(d, d) for d in h]), ())
        elif len(h) == 2 * self.ndim:
            halo = h
        else:
            raise ValueError(f"Invalid halo length {len(h)} for ndim={self.ndim}")
        if any(h < 0 for h in halo):
            raise ValueError("Halo widths must be non-negative")
        return halo

    def _build_topo(self) -> Tuple[MPI.Comm, Dict[Tuple[str, int], int]]:
        """Create the Cartesian communicator and map neighboring ranks on the distribution axis."""
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

    def _calc_local_dims(self) -> tuple:
        """Compute this rank's local block shape before halo padding."""
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

    def _calc_local_extent(self) -> tuple:
        """Compute this rank's local block shape after halo padding."""
        ext = []
        for ax in range(self.ndim):
            minus_halo, plus_halo = self.halo[2 * ax], self.halo[2 * ax + 1]
            ext.append(self.local_dims[ax] + minus_halo + plus_halo)
        return tuple(ext)

    def _validate_local_array_shape(
        self, x: DistributedArray, expected_shape: tuple, name: str
    ) -> None:
        """Raise if a distributed input does not match this rank's expected local shape."""
        local_shapes = self.cart_comm.allgather(
            (x.local_array.size, int(np.prod(expected_shape)), expected_shape)
        )
        for rank, (actual_size, expected_size, shape) in enumerate(local_shapes):
            if actual_size != expected_size:
                raise ValueError(
                    "MPIHalo input local shapes do not match the Cartesian block "
                    f"decomposition: rank {rank}: {name} local array has size "
                    f"{actual_size}, expected {expected_size} for local shape {shape}"
                )

    def _validate_exchange_widths(self) -> None:
        """
        Raise if the requested halos cannot be exchanged with one-hop neighbors.
        For example:
            - Halo width Larger than local block size or that of the remote neighbors.
        """
        width_error = 1
        mismatch_error = 2
        local_error = 0
        for ax in range(self.ndim):
            before, after = self.halo[2 * ax], self.halo[2 * ax + 1]
            minus_nbr, plus_nbr = self.neigh[("-", ax)], self.neigh[("+", ax)]
            local_dim = self.local_dims[ax]

            if before > local_dim and minus_nbr != MPI.PROC_NULL:
                local_error |= width_error
            if after > local_dim and plus_nbr != MPI.PROC_NULL:
                local_error |= width_error

            plus_neighbor_before = self.cart_comm.sendrecv(
                before, dest=minus_nbr, source=plus_nbr
            )
            minus_neighbor_after = self.cart_comm.sendrecv(after, dest=plus_nbr, source=minus_nbr)
            if plus_nbr != MPI.PROC_NULL and after != plus_neighbor_before:
                local_error |= mismatch_error
            if minus_nbr != MPI.PROC_NULL and before != minus_neighbor_after:
                local_error |= mismatch_error

        global_error = self.cart_comm.allreduce(local_error, op=MPI.BOR)
        if global_error & width_error:
            raise ValueError(
                "MPIHalo halo widths are not supported by the current one-hop "
                "exchange: halo width exceeds local block size"
            )
        if global_error & mismatch_error:
            raise ValueError(
                "MPIHalo halo widths are not supported by the current one-hop "
                "exchange: halo width does not match neighbor halo width"
            )

    def _exchange_along_axis(self, ncp: Any, arr: Any, axis: int, before: int, after: int, engine: str) -> None:
        """Exchange boundary/halo slices with neighboring ranks along one axis."""
        minus_nbr, plus_nbr = self.neigh[("-", axis)], self.neigh[("+", axis)]
        # slice definitions
        slicer = [slice(None)] * self.ndim
        # send before
        if before and minus_nbr != MPI.PROC_NULL:
            snd_s = slicer.copy()
            snd_s[axis] = slice(before, 2 * before)
            snd = arr[tuple(snd_s)].copy()
            rcv = ncp.empty_like(snd)
            rcv = self._sendrecv(
                self.cart_comm,
                None,
                snd,
                rcv,
                dest=minus_nbr,
                source=minus_nbr,
                engine=engine,
            )
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
            rcv = self._sendrecv(
                self.cart_comm,
                None,
                snd,
                rcv,
                dest=plus_nbr,
                source=plus_nbr,
                engine=engine,
            )
            arr[tuple(rcv_s)] = rcv

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(
                f"x should have partition={Partition.SCATTER} Got {x.partition} instead..."
            )

        self._validate_local_array_shape(x, self.local_dims, "x")

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
                ncp, halo_arr, axis=ax, before=before, after=after, engine=x.engine
            )

        y[:] = halo_arr.ravel()
        return y

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        if x.partition != Partition.SCATTER:
            raise ValueError(
                f"x should have partition={Partition.SCATTER} Got {x.partition} instead..."
            )
        self._validate_local_array_shape(x, self.local_extent, "x")

        res = DistributedArray(
            global_shape=self.shape[1],
            partition=Partition.SCATTER,
            local_shapes=self._local_dim_sizes,
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
