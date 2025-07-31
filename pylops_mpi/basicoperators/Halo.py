import math
import numpy as np
from enum import Enum
from mpi4py import MPI

from pylops.utils.backend import get_module
from pylops_mpi import MPILinearOperator, DistributedArray, Partition


class ScatterType(Enum):
    BLOCK = "BLOCK"
    SLAB   = "SLAB"
    STRIPE = "STRIPE"

class BoundaryType(Enum):
    ZERO     = "ZERO"
    REFLECT  = "REFLECT"
    PERIODIC = "PERIODIC"

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
        scatter: ScatterType = ScatterType.BLOCK,
        proc_grid_shape: tuple = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
        boundary_mode: BoundaryType = BoundaryType.ZERO,
        dtype = np.float64
    ):
        self.global_dims = tuple(dims)
        self.ndim = len(dims)
        self.halo = self._parse_halo(halo)
        self.scatter_type = scatter
        self.boundary_mode = boundary_mode
        self.comm  = comm
        self.dtype = dtype

        if self.scatter_type == ScatterType.BLOCK:
            proc_grid_shape = proc_grid_shape or tuple([1] * self.ndim)
        self.proc_grid_shape = tuple(proc_grid_shape)
        
        self.cart_comm, self.neigh = self._build_topo()
        self.local_dims   = self._calc_local_dims()
        self.local_extent = self._calc_local_extent()
        gext = self._calc_global_extent()
        self.shape = (int(np.prod(gext)), int(np.prod(self.global_dims)))
        super().__init__(shape=self.shape, dtype=np.dtype(dtype), base_comm=comm)


    def _parse_halo(self, h):
        if isinstance(h, int): return (h,) * (2 * self.ndim)
        h = tuple(h)
        if len(h) == 1             : return h * (2 * self.ndim)
        if len(h) == self.ndim     : return sum(tuple([(d, d) for d in h]), ())
        if len(h) == 2 * self.ndim : return h
        raise ValueError(f"Invalid halo length {len(h)} for ndim={self.ndim}")

    def _build_topo(self):
        periods = [self.boundary_mode == BoundaryType.PERIODIC] * self.ndim
        cart_comm = self.comm.Create_cart(self.proc_grid_shape, periods=periods, reorder=True)
        neigh = {}
        for ax in range(self.ndim):
            before, after = cart_comm.Shift(ax, 1)
            neigh[('-', ax)] = before
            neigh[('+', ax)] =  after
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

    def _calc_global_extent(self):
        ext = []
        for ax, gdim in enumerate(self.global_dims):
            minus_halo, plus_halo = self.halo[2 * ax], self.halo[2 * ax + 1]
            ext.append(gdim + self.proc_grid_shape[ax] * (minus_halo + plus_halo))
        return tuple(ext)


    def _apply_bc_along_axis(self, ncp, arr, axis):
        minus_halo, plus_halo = self.halo[2 * axis], self.halo[2 * axis + 1]
        slicer = [slice(None)]*self.ndim
        minus_nbr, plus_nbr = self.neigh[('-',axis)], self.neigh[('+',axis)]
        # before
        if minus_halo and minus_nbr == MPI.PROC_NULL:
            slicer_minus  = slicer.copy();
            slicer_minus[axis]  = slice(0, minus_halo)
            if self.boundary_mode == BoundaryType.ZERO:
                arr[tuple(slicer_minus)] = 0
            elif self.boundary_mode == BoundaryType.REFLECT:
                slicer_core_m = slicer.copy()
                slicer_core_m[axis] = slice(minus_halo, 2 * minus_halo)
                arr[tuple(slicer_minus)] = ncp.flip(arr[tuple(slicer_core_m)], axis=axis)
        # after
        if plus_halo and plus_nbr == MPI.PROC_NULL:
            slicer_plus   = slicer.copy(); slicer_plus[axis]   = slice(-plus_halo, None)
            if self.boundary_mode == BoundaryType.ZERO:
                arr[tuple(slicer_plus)] = 0
            elif self.boundary_mode == BoundaryType.REFLECT:
                slicer_core_p = slicer.copy();
                slicer_core_p[axis] = slice(-2 * plus_halo, -plus_halo)
                arr[tuple(slicer_plus)] = ncp.flip(arr[tuple(slicer_core_p)], axis=axis)

    def _exchange_along_axis(self, ncp, arr, axis, before, after):
        minus_nbr,plus_nbr = self.neigh[('-', axis)],  self.neigh[('+', axis)]
        # slice definitions
        slicer = [slice(None)] * self.ndim
        # send before
        if before and minus_nbr != MPI.PROC_NULL:
            snd_s = slicer.copy(); snd_s[axis] = slice(before, 2 * before)
            snd = arr[tuple(snd_s)].copy()
            rcv = ncp.empty_like(snd)
            self.cart_comm.Sendrecv(snd, dest=minus_nbr, recvbuf=rcv, source=minus_nbr)
            rcv_s = slicer.copy(); rcv_s[axis] = slice(0, before)
            arr[tuple(rcv_s)] = rcv
        # send after
        if after and plus_nbr != MPI.PROC_NULL:
            snd_s = slicer.copy(); snd_s[axis] = slice(-2 * after, -after)
            rcv_s = slicer.copy(); rcv_s[axis] = slice(-after, None)
            snd = arr[tuple(snd_s)].copy()
            rcv = ncp.empty_like(snd)
            self.cart_comm.Sendrecv(snd, dest=plus_nbr, recvbuf=rcv, source=plus_nbr)
            arr[tuple(rcv_s)] = rcv

    def _matvec(self, x):
        ncp = get_module(x.engine)
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER} Got {x.partition} instead...")

        core = x.local_array.reshape(self.local_dims)
        halo_arr = ncp.zeros(self.local_extent, dtype=self.dtype)
        # insert core
        core_slices = [slice(left, left + ldim) for left, ldim in zip(self.halo[::2], self.local_dims)]
        halo_arr[tuple(core_slices)] = core

        # exchange along each axis
        for ax in range(self.ndim):
            before, after = self.halo[2 * ax], self.halo[2 * ax + 1]
            self._exchange_along_axis(ncp, halo_arr, axis=ax, before=before, after=after)

        # apply BCs
        for ax in range(self.ndim):
            self._apply_bc_along_axis(ncp, halo_arr, axis=ax)
        # pack result
        res = DistributedArray(global_shape=self.shape[0],
                                partition=Partition.SCATTER)
        res[:] = halo_arr.ravel()
        return res

    def _rmatvec(self, x):
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER} Got {x.partition} instead...")
        res = DistributedArray(global_shape=self.shape[1],
                               partition=Partition.SCATTER)
        arr = x.local_array.reshape(self.local_extent)
        core_slices = [slice(left, left + ldim) for left, ldim in zip(self.halo[::2], self.local_dims)]
        core = arr[tuple(core_slices)]
        res[:] = core.ravel()
        return res