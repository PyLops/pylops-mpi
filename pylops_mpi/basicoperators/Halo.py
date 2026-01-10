import math
import numpy as np
from mpi4py import MPI

from pylops.utils.backend import get_module
from pylops_mpi import MPILinearOperator, DistributedArray, Partition


def local_block_split(global_shape: tuple, comm, grid_shape: tuple = None) -> tuple:
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


class MPIHalo(MPILinearOperator):
    def __init__(
        self,
        dims: tuple,
        halo,
        proc_grid_shape: tuple = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
        dtype=np.float64,
        normalize: bool = False,
    ):
        self.global_dims = tuple(dims)
        self.ndim = len(dims)

        self.normalize = normalize
        self.comm = comm
        self.dtype = dtype

        self.proc_grid_shape = tuple(proc_grid_shape)

        self.cart_comm, self.neigh = self._build_topo()
        self.halo = self._parse_halo(halo)

        self.local_dims = self._calc_local_dims()
        self.local_extent = self._calc_local_extent()
        self._local_extent_sizes = self.comm.allgather(int(np.prod(self.local_extent)))

        self.shape = (
            int(np.sum(self._local_extent_sizes)),
            int(np.prod(self.global_dims)),
        )
        self._norm_factors = self._compute_norm_factors() if self.normalize else None
        super().__init__(shape=self.shape, dtype=np.dtype(dtype), base_comm=comm)

    def _parse_halo(self, h):
        if isinstance(h, int):
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
        for ax, (gdim, coord, grid_procs) in enumerate(zip(self.global_dims, coords, self.proc_grid_shape)):
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

    def _compute_norm_factors(self):
        weights = np.ones(self.local_dims, dtype=np.float64)
        for ax in range(self.ndim):
            before, after = self.halo[2 * ax], self.halo[2 * ax + 1]
            if before == 0 and after == 0:
                continue
            n_local = self.local_dims[ax]
            factors = np.ones(n_local, dtype=np.float64)
            minus_nbr, plus_nbr = self.neigh[("-", ax)], self.neigh[("+", ax)]
            minus_copy = minus_nbr != MPI.PROC_NULL
            plus_copy = plus_nbr != MPI.PROC_NULL
            if before and minus_copy:
                factors[:before] += 1.0
            if after and plus_copy:
                factors[n_local - after:] += 1.0
            shape = [1] * self.ndim
            shape[ax] = n_local
            weights *= factors.reshape(shape)
        return (1.0 / np.sqrt(weights)).astype(self.dtype, copy=False)

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

    def _exchange_adjoint_along_axis(self, ncp, arr, axis, before, after):
        minus_nbr, plus_nbr = self.neigh[("-", axis)], self.neigh[("+", axis)]
        out = arr.copy()
        if before == 0 and after == 0:
            return out

        def axis_slice(start, end):
            s = [slice(None)] * self.ndim
            s[axis] = slice(start, end)
            return tuple(s)

        empty = axis_slice(0, 0)
        halo_minus = axis_slice(0, before) if before else None
        halo_plus = axis_slice(-after, None) if after else None
        core_minus = axis_slice(before, 2 * before) if before else None
        core_plus = axis_slice(-2 * after, -after) if after else None

        # remove contributions from halo slabs along this axis (forward overwrites them)
        if halo_minus:
            out[halo_minus] = 0
        if halo_plus:
            out[halo_plus] = 0

        # send right halo to plus neighbor, receive right halo from minus neighbor
        snd_slice = halo_plus or empty
        rcv_slice = halo_minus or empty
        rcv = ncp.zeros_like(arr[rcv_slice])
        self.cart_comm.Sendrecv(arr[snd_slice].copy(), dest=plus_nbr, recvbuf=rcv, source=minus_nbr)
        if core_minus:
            out[core_minus] += rcv

        # send left halo to minus neighbor, receive left halo from plus neighbor
        snd_slice = halo_minus or empty
        rcv_slice = halo_plus or empty
        rcv = ncp.zeros_like(arr[rcv_slice])
        self.cart_comm.Sendrecv(arr[snd_slice].copy(), dest=minus_nbr, recvbuf=rcv, source=plus_nbr)
        if core_plus:
            out[core_plus] += rcv
        return out

    def _matvec(self, x):
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER} Got {x.partition} instead...")

        y = DistributedArray(
            global_shape=self.shape[0],
            partition=Partition.SCATTER,
            local_shapes=self._local_extent_sizes,
        )

        core = x.local_array.reshape(self.local_dims)
        if self.normalize:
            norm = self._norm_factors if ncp is np else ncp.asarray(self._norm_factors)
            core = core * norm
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
            self._exchange_along_axis(ncp, halo_arr, axis=ax, before=before, after=after)

        y[:] = halo_arr.ravel()
        return y

    def _rmatvec(self, x):
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER} Got {x.partition} instead...")

        y = DistributedArray(global_shape=self.shape[1], partition=Partition.SCATTER)

        arr = x.local_array.reshape(self.local_extent).copy()
        # adjoint of halo exchange (reverse axis order because of the Transpose)
        for ax in reversed(range(self.ndim)):
            before, after = self.halo[2 * ax], self.halo[2 * ax + 1]
            arr = self._exchange_adjoint_along_axis(ncp, arr, axis=ax, before=before, after=after)
        core_slices = [
            slice(left, left + ldim)
            for left, ldim in zip(self.halo[::2], self.local_dims)
        ]
        core = arr[tuple(core_slices)]
        if self.normalize:
            norm = self._norm_factors if ncp is np else ncp.asarray(self._norm_factors)
            core = core * norm

        y[:] = core.ravel()
        return y
