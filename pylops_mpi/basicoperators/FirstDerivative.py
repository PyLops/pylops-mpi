from typing import Callable, Union
import numpy as np
from mpi4py import MPI

from pylops.utils.backend import get_module
from pylops.utils.typing import DTypeLike, InputDimsLike
from pylops.utils._internal import _value_or_sized_to_tuple

from pylops_mpi import (
    DistributedArray,
    MPILinearOperator,
    Partition
)

from pylops_mpi.utils.decorators import reshaped


class MPIFirstDerivative(MPILinearOperator):
    r"""MPI First Derivative

    Apply a first derivative using a multiple-point stencil finite-difference
    approximation with :class:`pylops_mpi.DistributedArray`. The First-Derivative
    is calculated along ``axis=0``.

    Parameters
    ----------
    dims : :obj:`int` or :obj:`tuple`
        Number of samples for each dimension.
    sampling : :obj:`float`, optional
        Sampling step :math:`\Delta x`.
    kind : :obj:`str`, optional
        Derivative kind (``forward``, ``centered``, or ``backward``).
    edge : :obj:`bool`, optional
        Use reduced order derivative at edges (``True``) or
        ignore them (``False``). This is currently only available
        for centered derivative.
    order : :obj:`int`, optional
        Derivative order (``3`` or ``5``). This is currently only available
        for centered derivative.
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        MPI Base Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
    dtype : :obj:`str`, optional
        Type of elements in the input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape

    Notes
    -----
    The MPIFirstDerivative operator applies a first derivative to a :class:`pylops_mpi.DistributedArray`
    using either a first-order backward and forward stencil or a second or third ordered centered stencil.

    When computing the first derivative using a :class:`pylops_mpi.DistributedArray`, a technique named **ghosting**
    is employed, often in conjunction with MPI for communication between different processes. Ghosting involves
    creating additional **"ghost cells"** around the boundary of each process's local data domain. These ghost cells
    are replicated copies of the border cells from neighboring processes, these cells allow each process
    to perform local computations without the need for explicit communication with other processes.

    Now, for a one-dimensional DistributedArray, the first order forward stencil is:

    .. math::
        y[i] = (x[i+1] - x[i]) / \Delta x

    while the first order backward stencil is:

    .. math::
        y[i] = (x[i] - x[i-1]) / \Delta x

    and the second-order centered stencil is:

    .. math::
        y[i] = 0.5 * (x[i+1] - x[i-1]) / \Delta x

    where :math:`y` is a DistributedArray and :math:`x` is a Ghosted array created by adding
    border cells of neighboring processes to the local array at each rank.

    Formulas for the third-order centered stencil can be found at
    this `link <https://en.wikipedia.org/wiki/Finite_difference_coefficient>`_.

    """

    def __init__(self,
                 dims: Union[int, InputDimsLike],
                 sampling: float = 1.0,
                 kind: str = "centered",
                 edge: bool = False,
                 order: int = 3,
                 base_comm: MPI.Comm = MPI.COMM_WORLD,
                 dtype: DTypeLike = np.float64):
        self.dims = _value_or_sized_to_tuple(dims)
        shape = (int(np.prod(dims)),) * 2
        super().__init__(shape=shape, dtype=np.dtype(dtype), base_comm=base_comm)
        self.sampling = sampling
        self.kind = kind
        self.edge = edge
        self.order = order
        self._register_multiplications(self.kind, self.order)

    def _register_multiplications(
            self,
            kind: str,
            order: int,
    ) -> None:
        # choose _matvec and _rmatvec kind
        self._hmatvec: Callable
        self._hrmatvec: Callable
        if kind == "forward":
            self._hmatvec = self._matvec_forward
            self._hrmatvec = self._rmatvec_forward
        elif kind == "centered":
            if order == 3:
                self._hmatvec = self._matvec_centered3
                self._hrmatvec = self._rmatvec_centered3
            elif order == 5:
                self._hmatvec = self._matvec_centered5
                self._hrmatvec = self._rmatvec_centered5
            else:
                raise NotImplementedError("'order' must be '3, or '5'")
        elif kind == "backward":
            self._hmatvec = self._matvec_backward
            self._hrmatvec = self._rmatvec_backward
        else:
            raise NotImplementedError(
                "'kind' must be 'forward', 'centered', or 'backward'"
            )

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        # If Partition.BROADCAST, then convert to Partition.SCATTER
        if x.partition is Partition.BROADCAST:
            x = DistributedArray.to_dist(x=x.local_array)
        return self._hmatvec(x)

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        # If Partition.BROADCAST, then convert to Partition.SCATTER
        if x.partition is Partition.BROADCAST:
            x = DistributedArray.to_dist(x=x.local_array)
        return self._hrmatvec(x)

    @reshaped
    def _matvec_forward(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        y = DistributedArray(global_shape=x.global_shape, local_shapes=x.local_shapes, 
                             axis=x.axis, engine=x.engine, dtype=self.dtype)
        ghosted_x = x.add_ghost_cells(cells_back=1)
        y_forward = ghosted_x[1:] - ghosted_x[:-1]
        if self.rank == self.size - 1:
            y_forward = ncp.append(y_forward, ncp.zeros((1,) + self.dims[1:]), axis=0)
        y[:] = y_forward / self.sampling
        return y

    @reshaped
    def _rmatvec_forward(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        y = DistributedArray(global_shape=x.global_shape, local_shapes=x.local_shapes, 
                             axis=x.axis, engine=x.engine, dtype=self.dtype)
        y[:] = 0
        if self.rank == self.size - 1:
            y[:-1] -= x[:-1]
        else:
            y[:] -= x[:]
        ghosted_x = x.add_ghost_cells(cells_front=1)
        y_forward = ghosted_x[:-1]
        if self.rank == 0:
            y_forward = ncp.append(ncp.zeros((1,) + self.dims[1:]), y_forward, axis=0)
        y[:] += y_forward
        y[:] /= self.sampling
        return y

    @reshaped
    def _matvec_backward(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        y = DistributedArray(global_shape=x.global_shape, local_shapes=x.local_shapes, 
                             axis=x.axis, engine=x.engine, dtype=self.dtype)
        ghosted_x = x.add_ghost_cells(cells_front=1)
        y_backward = ghosted_x[1:] - ghosted_x[:-1]
        if self.rank == 0:
            y_backward = ncp.append(ncp.zeros((1,) + self.dims[1:]), y_backward, axis=0)
        y[:] = y_backward / self.sampling
        return y

    @reshaped
    def _rmatvec_backward(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        y = DistributedArray(global_shape=x.global_shape, local_shapes=x.local_shapes, 
                             axis=x.axis, engine=x.engine, dtype=self.dtype)
        y[:] = 0
        ghosted_x = x.add_ghost_cells(cells_back=1)
        y_backward = ghosted_x[1:]
        if self.rank == self.size - 1:
            y_backward = ncp.append(y_backward, ncp.zeros((1,) + self.dims[1:]), axis=0)
        y[:] -= y_backward
        if self.rank == 0:
            y[1:] += x[1:]
        else:
            y[:] += x[:]
        y[:] /= self.sampling
        return y

    @reshaped
    def _matvec_centered3(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        y = DistributedArray(global_shape=x.global_shape, local_shapes=x.local_shapes, 
                             axis=x.axis, engine=x.engine, dtype=self.dtype)
        ghosted_x = x.add_ghost_cells(cells_front=1, cells_back=1)
        y_centered = 0.5 * (ghosted_x[2:] - ghosted_x[:-2])
        if self.rank == 0:
            y_centered = ncp.append(ncp.zeros((1,) + self.dims[1:]), y_centered, axis=0)
        if self.rank == self.size - 1:
            y_centered = ncp.append(y_centered, ncp.zeros((min(y.global_shape[0] - 1, 1), ) + self.dims[1:]), axis=0)
        y[:] = y_centered
        if self.edge:
            if self.rank == 0:
                y[0] = x[1] - x[0]
            if self.rank == self.size - 1:
                y[-1] = x[-1] - x[-2]
        y[:] /= self.sampling
        return y

    @reshaped
    def _rmatvec_centered3(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        y = DistributedArray(global_shape=x.global_shape, local_shapes=x.local_shapes, 
                             axis=x.axis, engine=x.engine, dtype=self.dtype)
        y[:] = 0

        ghosted_x = x.add_ghost_cells(cells_back=2)
        y_centered = 0.5 * ghosted_x[1:-1]
        if self.rank == self.size - 1:
            y_centered = ncp.append(y_centered, ncp.zeros((min(y.global_shape[0], 2),) + self.dims[1:]), axis=0)
        y[:] -= y_centered

        ghosted_x = x.add_ghost_cells(cells_front=2)
        y_centered = 0.5 * ghosted_x[1:-1]
        if self.rank == 0:
            y_centered = ncp.append(ncp.zeros((min(y.global_shape[0], 2),) + self.dims[1:]), y_centered, axis=0)
        y[:] += y_centered
        if self.edge:
            if self.rank == 0:
                y[0] -= x[0]
                y[1] += x[0]
            if self.rank == self.size - 1:
                y[-2] -= x[-1]
                y[-1] += x[-1]
        y[:] /= self.sampling
        return y

    @reshaped
    def _matvec_centered5(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        y = DistributedArray(global_shape=x.global_shape, local_shapes=x.local_shapes, 
                             axis=x.axis, engine=x.engine, dtype=self.dtype)
        ghosted_x = x.add_ghost_cells(cells_front=2, cells_back=2)
        y_centered = (
            ghosted_x[:-4] / 12.0
            - 2 * ghosted_x[1:-3] / 3.0
            + 2 * ghosted_x[3:-1] / 3.0
            - ghosted_x[4:] / 12.0
        )
        if self.rank == 0:
            y_centered = ncp.append(ncp.zeros((min(y.global_shape[0], 2),) + self.dims[1:]), y_centered, axis=0)
        if self.rank == self.size - 1:
            y_centered = ncp.append(y_centered, ncp.zeros((min(y.global_shape[0] - 2, 2),) + self.dims[1:]), axis=0)
        y[:] = y_centered
        if self.edge:
            if self.rank == 0:
                y[0] = x[1] - x[0]
                y[1] = 0.5 * (x[2] - x[0])
            if self.rank == self.size - 1:
                y[-1] = x[-1] - x[-2]
                y[-2] = 0.5 * (x[-1] - x[-3])
        y[:] /= self.sampling
        return y

    @reshaped
    def _rmatvec_centered5(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        y = DistributedArray(global_shape=x.global_shape, local_shapes=x.local_shapes, 
                             axis=x.axis, engine=x.engine, dtype=self.dtype)
        y[:] = 0
        ghosted_x = x.add_ghost_cells(cells_back=4)
        y_centered = ghosted_x[2:-2] / 12.0
        if self.rank == self.size - 1:
            y_centered = ncp.append(y_centered, ncp.zeros((min(y.global_shape[0], 4),) + self.dims[1:]), axis=0)
        y[:] += y_centered

        ghosted_x = x.add_ghost_cells(cells_front=1, cells_back=3)
        y_centered = 2.0 * ghosted_x[2:-2] / 3.0
        if self.rank == 0:
            y_centered = ncp.append(ncp.zeros((1,) + self.dims[1:]), y_centered, axis=0)
        if self.rank == self.size - 1:
            y_centered = ncp.append(y_centered, ncp.zeros((min(y.global_shape[0] - 1, 3),) + self.dims[1:]), axis=0)
        y[:] -= y_centered

        ghosted_x = x.add_ghost_cells(cells_front=3, cells_back=1)
        y_centered = 2.0 * ghosted_x[2:-2] / 3.0
        if self.rank == 0:
            y_centered = ncp.append(ncp.zeros((min(y.global_shape[0], 3),) + self.dims[1:]), y_centered, axis=0)
        if self.rank == self.size - 1:
            y_centered = ncp.append(y_centered, ncp.zeros((min(y.global_shape[0] - 3, 1),) + self.dims[1:]), axis=0)
        y[:] += y_centered

        ghosted_x = x.add_ghost_cells(cells_front=4)
        y_centered = ghosted_x[2:-2] / 12.0
        if self.rank == 0:
            y_centered = ncp.append(ncp.zeros((min(y.global_shape[0], 4),) + self.dims[1:]), y_centered, axis=0)
        y[:] -= y_centered
        if self.edge:
            if self.rank == 0:
                y[0] -= x[0] + 0.5 * x[1]
                y[1] += x[0]
                y[2] += 0.5 * x[1]
            if self.rank == self.size - 1:
                y[-3] -= 0.5 * x[-2]
                y[-2] -= x[-1]
                y[-1] += 0.5 * x[-2] + x[-1]
        y[:] /= self.sampling
        return y
