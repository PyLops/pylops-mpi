import numpy as np

from typing import Callable, Union
from pylops.utils.typing import DTypeLike, InputDimsLike
from pylops.utils._internal import _value_or_sized_to_tuple

from pylops_mpi import DistributedArray, MPILinearOperator, Partition
from pylops_mpi.utils.decorators import reshaped


class MPISecondDerivative(MPILinearOperator):
    r"""MPI Second Derivative

    Apply a second derivative using a three-point stencil finite-difference
    approximation with :class:`pylops_mpi.DistributedArray`. The Second Derivative
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
    dtype : :obj:`str`, optional
        Type of elements in the input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape

    Notes
    -----
    The MPISecondDerivative operator applies a second derivative to a :class:`pylops_mpi.DistributedArray`
    using either a second-order forward, backward or a centered stencil.

    Now, for a one-dimensional DistributedArray, the second-order forward stencil is:

    .. math::
        y[i] = (x[i+2] - 2 * x[i+1] + x[i]) / \mathbf{\Delta x}^2

    while the second-order backward stencil is:

    .. math::
        y[i] = (x[i] - 2 * x[i-1] + x[i-2]) / \mathbf{\Delta x}^2

    and the second-order centered stencil is:

    .. math::
        y[i] = (x[i+1] - 2 * x[i] + x[i-1]) / \mathbf{\Delta x}^2

    """

    def __init__(
            self,
            dims: Union[int, InputDimsLike],
            sampling: float = 1.0,
            kind: str = "centered",
            edge: bool = False,
            dtype: DTypeLike = np.float64,
    ) -> None:
        self.dims = _value_or_sized_to_tuple(dims)
        shape = (int(np.prod(dims)),) * 2
        super().__init__(shape=shape, dtype=np.dtype(dtype))
        self.sampling = sampling
        self.kind = kind
        self.edge = edge
        self._register_multiplications(self.kind)

    def _register_multiplications(
            self,
            kind: str,
    ) -> None:
        # choose _matvec and _rmatvec kind
        self._hmatvec: Callable
        self._hrmatvec: Callable
        if kind == "forward":
            self._hmatvec = self._matvec_forward
            self._hrmatvec = self._rmatvec_forward
        elif kind == "centered":
            self._hmatvec = self._matvec_centered
            self._hrmatvec = self._rmatvec_centered
        elif kind == "backward":
            self._hmatvec = self._matvec_backward
            self._hrmatvec = self._rmatvec_backward
        else:
            raise NotImplementedError(
                "'kind' must be 'forward', 'centered' or 'backward'"
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
        y = DistributedArray(global_shape=x.global_shape, dtype=self.dtype, axis=x.axis)
        ghosted_x = x.add_ghost_cells(cells_back=2)
        y_forward = ghosted_x[2:] - 2 * ghosted_x[1:-1] + ghosted_x[:-2]
        if self.rank == self.size - 1:
            y_forward = np.append(y_forward, np.zeros((min(y.global_shape[0], 2),) + self.dims[1:]), axis=0)
        y[:] = y_forward / self.sampling ** 2
        return y

    @reshaped
    def _rmatvec_forward(self, x: DistributedArray) -> DistributedArray:
        y = DistributedArray(global_shape=x.global_shape, dtype=self.dtype, axis=x.axis)
        y[:] = 0
        if self.rank == self.size - 1:
            y[:-2] += x[:-2]
        else:
            y[:] += x[:]

        ghosted_x = x.add_ghost_cells(cells_front=1, cells_back=1)
        y_forward = ghosted_x[:-2]
        if self.rank == 0:
            y_forward = np.insert(y_forward, 0, np.zeros((1,) + self.dims[1:]), axis=0)
        if self.rank == self.size - 1:
            y_forward = np.append(y_forward, np.zeros((min(1, y.global_shape[0] - 1),) + self.dims[1:]), axis=0)
        y[:] -= 2 * y_forward

        ghosted_x = x.add_ghost_cells(cells_front=2)
        y_forward = ghosted_x[:-2]
        if self.rank == 0:
            y_forward = np.insert(y_forward, 0, np.zeros((min(y.global_shape[0], 2),) + self.dims[1:]), axis=0)
        y[:] += y_forward
        y[:] /= self.sampling ** 2
        return y

    @reshaped
    def _matvec_backward(self, x: DistributedArray) -> DistributedArray:
        y = DistributedArray(global_shape=x.global_shape, dtype=self.dtype, axis=x.axis)
        ghosted_x = x.add_ghost_cells(cells_front=2)
        y_backward = ghosted_x[2:] - 2 * ghosted_x[1:-1] + ghosted_x[:-2]
        if self.rank == 0:
            y_backward = np.insert(y_backward, 0, np.zeros((min(y.global_shape[0], 2),) + self.dims[1:]), axis=0)
        y[:] = y_backward / self.sampling ** 2
        return y

    @reshaped
    def _rmatvec_backward(self, x: DistributedArray) -> DistributedArray:
        y = DistributedArray(global_shape=x.global_shape, dtype=self.dtype, axis=x.axis)
        y[:] = 0
        ghosted_x = x.add_ghost_cells(cells_back=2)
        y_backward = ghosted_x[2:]
        if self.rank == self.size - 1:
            y_backward = np.append(y_backward, np.zeros((min(2, y.global_shape[0]),) + self.dims[1:]), axis=0)
        y[:] += y_backward

        ghosted_x = x.add_ghost_cells(cells_front=1, cells_back=1)
        y_backward = 2 * ghosted_x[2:]
        if self.rank == 0:
            y_backward = np.insert(y_backward, 0, np.zeros((1,) + self.dims[1:]), axis=0)
        if self.rank == self.size - 1:
            y_backward = np.append(y_backward, np.zeros((min(1, y.global_shape[0] - 1),) + self.dims[1:]), axis=0)
        y[:] -= y_backward

        if self.rank == 0:
            y[2:] += x[2:]
        else:
            y[:] += x[:]
        y[:] /= self.sampling ** 2
        return y

    @reshaped
    def _matvec_centered(self, x: DistributedArray) -> DistributedArray:
        y = DistributedArray(global_shape=x.global_shape, dtype=self.dtype, axis=x.axis)
        ghosted_x = x.add_ghost_cells(cells_front=1, cells_back=1)
        y_centered = ghosted_x[2:] - 2 * ghosted_x[1:-1] + ghosted_x[:-2]
        if self.rank == 0:
            y_centered = np.insert(y_centered, 0, np.zeros((1,) + self.dims[1:]), axis=0)
        if self.rank == self.size - 1:
            y_centered = np.append(y_centered, np.zeros((min(1, y.global_shape[0] - 1),) + self.dims[1:]), axis=0)
        y[:] = y_centered
        if self.edge:
            if self.rank == 0:
                y[0] = x[0] - 2 * x[1] + x[2]
            if self.rank == self.size - 1:
                y[-1] = x[-3] - 2 * x[-2] + x[-1]
        y[:] /= self.sampling ** 2
        return y

    @reshaped
    def _rmatvec_centered(self, x: DistributedArray) -> DistributedArray:
        y = DistributedArray(global_shape=x.global_shape, dtype=self.dtype, axis=x.axis)
        y[:] = 0
        ghosted_x = x.add_ghost_cells(cells_back=2)
        y_centered = ghosted_x[1:-1]
        if self.rank == self.size - 1:
            y_centered = np.append(y_centered, np.zeros((min(2, y.global_shape[0]),) + self.dims[1:]), axis=0)
        y[:] += y_centered

        ghosted_x = x.add_ghost_cells(cells_front=1, cells_back=1)
        y_centered = 2 * ghosted_x[1:-1]
        if self.rank == 0:
            y_centered = np.insert(y_centered, 0, np.zeros((1,) + self.dims[1:]), axis=0)
        if self.rank == self.size - 1:
            y_centered = np.append(y_centered, np.zeros((min(1, y.global_shape[0] - 1),) + self.dims[1:]), axis=0)
        y[:] -= y_centered

        ghosted_x = x.add_ghost_cells(cells_front=2)
        y_centered = ghosted_x[1:-1]
        if self.rank == 0:
            y_centered = np.insert(y_centered, 0, np.zeros((min(2, y.global_shape[0]),) + self.dims[1:]), axis=0)
        y[:] += y_centered
        if self.edge:
            if self.rank == 0:
                y[0] += x[0]
                y[1] -= 2 * x[0]
                y[2] += x[0]
            if self.rank == self.size - 1:
                y[-3] += x[-1]
                y[-2] -= 2 * x[-1]
                y[-1] += x[-1]
        y[:] /= self.sampling ** 2
        return y
