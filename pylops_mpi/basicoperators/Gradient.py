from typing import Union
from mpi4py import MPI
import numpy as np

from pylops import FirstDerivative
from pylops.utils import InputDimsLike, DTypeLike
from pylops.utils._internal import _value_or_sized_to_tuple

from pylops_mpi import MPIStackedLinearOperator
from pylops_mpi.basicoperators.VStack import MPIStackedVStack
from pylops_mpi.basicoperators.BlockDiag import MPIBlockDiag
from pylops_mpi.basicoperators.FirstDerivative import MPIFirstDerivative
from pylops_mpi.DistributedArray import (
    DistributedArray,
    StackedDistributedArray,
    Partition,
    local_split
)


class MPIGradient(MPIStackedLinearOperator):
    r"""MPI Gradient

    Apply gradient operator to a multi-dimensional distributed
    array.

    .. note:: At least 2 dimensions are required, use
        :py:func:`pylops_mpi.MPIFirstDerivative` for 1d arrays.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension.
    sampling : :obj:`tuple`, optional
        Sampling steps for each direction.
    edge : :obj:`bool`, optional
        Use reduced order derivative at edges (``True``) or ignore them (``False``).
    kind : :obj:`str`, optional
        Derivative kind (``forward``, ``centered``, or ``backward``).
    base_comm : : obj:`MPI.Comm`, optional
        MPI Base Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Notes
    -----
    The MPIGradient operator applies a first-order derivative to each dimension of
    a multi-dimensional distributed array in forward mode.

    We utilize the :py:class:`pylops_mpi.basicoperators.MPIFirstDerivative` to
    calculate the first derivative along the first direction(i.e., axis=0).
    For other values of axis, the :py:class:`pylops.FirstDerivative` operator is
    pushed into the :py:class:`pylops_mpi.basicoperators.MPIBlockDiag` operator.

    Finally, using the :py:class:`pylops_mpi.basicoperators.MPIStackedVStack` we vertically
    stack the MPIFirstDerivative and the MPIBlockDiag operators.

    For the forward mode, the matrix vector product is performed between the
    :py:class:`pylops_mpi.basicoperators.MPIStackedVStack` and the :py:class:`pylops_mpi.DistributedArray`.

    For simplicity, given a three-dimensional array, the MPIGradient in forward mode using a
    centered stencil can be expressed as:

    .. math::
        \mathbf{g}_{i, j, k} =
            (f_{i+1, j, k} - f_{i-1, j, k}) / d_1 \mathbf{i_1} +
            (f_{i, j+1, k} - f_{i, j-1, k}) / d_2 \mathbf{i_2} +
            (f_{i, j, k+1} - f_{i, j, k-1}) / d_3 \mathbf{i_3}

    In adjoint mode, the adjoint matrix vector product is performed between the
    :py:class:`pylops_mpi.basicoperators.MPIStackedVStack` and the :py:class:`pylops_mpi.StackedDistributedArray`.

    """

    def __init__(self,
                 dims: Union[int, InputDimsLike],
                 sampling: int = 1,
                 edge: bool = False,
                 kind: str = "centered",
                 base_comm: MPI.Comm = MPI.COMM_WORLD,
                 dtype: DTypeLike = "float64",
                 ):
        self.dims = _value_or_sized_to_tuple(dims)
        ndims = len(self.dims)
        sampling = _value_or_sized_to_tuple(sampling, repeat=ndims)
        self.sampling = sampling
        self.edge = edge
        self.kind = kind
        self.base_comm = base_comm
        self.dtype = np.dtype(dtype)
        self.Op = self._calc_stack_op(ndims)
        super().__init__(shape=self.Op.shape, dtype=dtype, base_comm=base_comm)

    def _matvec(self, x: DistributedArray) -> StackedDistributedArray:
        return self.Op @ x

    def _rmatvec(self, x: StackedDistributedArray) -> DistributedArray:
        return self.Op.H @ x

    def _calc_stack_op(self, ndims):
        local_dims = local_split(tuple(self.dims), self.base_comm, Partition.SCATTER, axis=0)
        grad_ops = []
        Op1 = MPIFirstDerivative(dims=self.dims, sampling=self.sampling[0],
                                 kind=self.kind, edge=self.edge,
                                 dtype=self.dtype)
        grad_ops.append(Op1)
        for iax in range(1, ndims):
            diag = MPIBlockDiag([
                FirstDerivative(
                    dims=local_dims,
                    axis=iax,
                    sampling=self.sampling[iax],
                    edge=self.edge,
                    kind=self.kind,
                    dtype=self.dtype)
            ])
            grad_ops.append(diag)
        return MPIStackedVStack(ops=grad_ops)
