from typing import Tuple
import numpy as np
from mpi4py import MPI

from pylops.utils.typing import DTypeLike, InputDimsLike
from pylops.basicoperators import SecondDerivative
from pylops.utils.backend import get_normalize_axis_index

from pylops_mpi import DistributedArray, MPILinearOperator, Partition
from pylops_mpi.DistributedArray import local_split
from pylops_mpi.basicoperators import MPIBlockDiag, MPISecondDerivative


class MPILaplacian(MPILinearOperator):
    r"""MPI Laplacian

    Apply second-order centered Laplacian operator to a multi-dimensional
    distributed array.

    .. note:: At least 2 dimensions are required, use
      :py:class:`pylops_mpi.basicoperators.MPISecondDerivative` for one dimension.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension.
    axes : :obj:`int`, optional
        Axes along which the Laplacian is applied.
    weights : :obj:`tuple`, optional
        Weight to apply to each direction (real laplacian operator if
        ``weights=(1, 1)``)
    sampling : :obj:`tuple`, optional
        Sampling steps for each direction
    edge : :obj:`bool`, optional
        Use reduced order derivative at edges (``True``) or
        ignore them (``False``) for centered derivative
    kind : :obj:`str`, optional
        Derivative kind (``forward``, ``centered``, or ``backward``)
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        MPI Base Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Raises
    ------
    ValueError
        If ``axes``. ``weights``, and ``sampling`` do not have the same size.

    Notes
    -----
    The MPILaplacian operator applies a second derivative along multiple directions of
    a multi-dimensional distributed array.

    We utilize the :py:class:`pylops_mpi.basicoperators.MPISecondDerivative` to
    calculate the second derivative along the first direction(i.e., axis=0).
    For other values of axis, the :py:class:`pylops.SecondDerivative` operator is
    pushed into the :py:class:`pylops_mpi.basicoperators.MPIBlockDiag` operator.
    Subsequently, the matrix-vector product is performed between the
    SecondDerivative operator and the distributed data.

    For simplicity, given a two-dimensional array, the Laplacian is:

    .. math::
        y[i, j] = (x[i+1, j] + x[i-1, j] + x[i, j-1] +x[i, j+1] - 4x[i, j])
                  / (\Delta x \Delta y)

    """

    def __init__(self, dims: InputDimsLike,
                 axes: InputDimsLike = (-2, -1),
                 weights: Tuple[float, ...] = (1, 1),
                 sampling: Tuple[float, ...] = (1, 1),
                 edge: bool = False,
                 kind: str = "centered",
                 base_comm: MPI.Comm = MPI.COMM_WORLD,
                 dtype: DTypeLike = np.float64):
        self.dims = dims
        axes = tuple(get_normalize_axis_index()(ax, len(dims)) for ax in axes)
        if not (len(axes) == len(weights) == len(sampling)):
            raise ValueError("axes, weights, and sampling have different size")
        self.axes = axes
        self.weights = weights
        self.sampling = sampling
        self.edge = edge
        self.kind = kind
        self.dtype = np.dtype(dtype)
        self.base_comm = base_comm
        self.Op = self._calc_l2op()
        super().__init__(shape=self.Op.shape, dtype=self.dtype, base_comm=self.base_comm)

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        return self.Op @ x

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        return self.Op.H @ x

    def _calc_l2op(self):
        local_dims = local_split(tuple(self.dims), self.base_comm, Partition.SCATTER, axis=0)
        if self.axes[0] == 0:
            l2op = self.weights[0] * MPISecondDerivative(dims=self.dims,
                                                         sampling=self.sampling[0],
                                                         kind=self.kind,
                                                         edge=self.edge,
                                                         dtype=self.dtype)
        else:
            l2op = self.weights[0] * MPIBlockDiag(ops=[SecondDerivative(dims=local_dims,
                                                                        axis=self.axes[0],
                                                                        sampling=self.sampling[0],
                                                                        kind=self.kind,
                                                                        edge=self.edge,
                                                                        dtype=self.dtype)])
        for ax, samp, weight in zip(self.axes[1:], self.sampling[1:], self.weights[1:]):
            if ax == 0:
                l2op += weight * MPISecondDerivative(dims=self.dims,
                                                     sampling=samp,
                                                     kind=self.kind,
                                                     edge=self.edge,
                                                     dtype=self.dtype)
            else:
                l2op += weight * MPIBlockDiag(ops=[SecondDerivative(dims=local_dims,
                                                                    axis=ax,
                                                                    sampling=samp,
                                                                    kind=self.kind,
                                                                    edge=self.edge,
                                                                    dtype=self.dtype)])
        return l2op
