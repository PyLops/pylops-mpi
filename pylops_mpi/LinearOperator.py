import numpy as np
from mpi4py import MPI
from typing import Callable

from scipy.sparse._sputils import isintlike
from scipy.sparse.linalg._interface import _get_dtype

from pylops import LinearOperator
from pylops.utils import DTypeLike, ShapeLike

from pylops_mpi import DistributedArray


class MPILinearOperator(LinearOperator):
    """Common interface for performing matrix-vector products in distributed fashion.

    This class extends the :class:`pylops.LinearOperator`.
    This class provides methods to perform matrix-vector product and adjoint matrix-vector
    products using MPI.

    .. note:: End users of pylops-mpi should not use this class directly but simply
      use operators that are already implemented. This class is meant for
      developers only, it has to be used as the parent class of any new operator
      developed within pylops-mpi.

    Parameters
    ----------
    shape : :obj:`tuple(int, int)`
        Shape of the MPI Linear Operator.
    dtype : :obj:`str`
        Type of elements in input array.
    Op : :obj:`pylops.LinearOperator`, optional
        Linear Operator. Defaults to ``None``.
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        MPI Base Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.

    """

    def __init__(self, shape: ShapeLike, dtype: DTypeLike, Op: LinearOperator = None,
                 base_comm: MPI.Comm = MPI.COMM_WORLD):
        self.Op = Op
        # For MPI
        self.base_comm = base_comm
        self.size = self.base_comm.Get_size()
        self.rank = self.base_comm.Get_rank()
        super().__init__(Op=Op, dtype=dtype, shape=shape)

    def matvec(self, x: DistributedArray) -> DistributedArray:
        if self.Op:
            y = DistributedArray(global_shape=self.base_comm.allreduce(self.shape[1]), dtype=x.dtype)
            if isinstance(x, DistributedArray):
                x = x.local_array
            y[:] = self.Op._matvec(x)
        else:
            y = self._matvec(x)
        return y

    def rmatvec(self, x: DistributedArray) -> DistributedArray:
        if self.Op:
            y = DistributedArray(global_shape=self.base_comm.allreduce(self.shape[1]), dtype=x.dtype)
            if isinstance(x, DistributedArray):
                x = x.local_array
            y[:] = self.Op._rmatvec(x)
        else:
            y = self._rmatvec(x)
        return y

    def dot(self, x: DistributedArray):
        if isinstance(x, MPILinearOperator):
            Op = _ProductLinearOperator(self, x)
            Op.clinear = (getattr(self, 'clinear', True)
                          and getattr(x, 'clinear', True))
            return Op
        elif np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            if x is None or x.ndim == 1:
                return self.matvec(x)
            elif x.ndim == 2:
                return self.matmat(x)
            else:
                raise ValueError('expected 1-d or 2-d array or matrix, got %r'
                                 % x)

    def adjoint(self):
        return self._adjoint()

    H = property(adjoint)

    def transpose(self):
        return self._transpose()

    T = property(transpose)

    def __mul__(self, x):
        return self.dot(x)

    def __rmul__(self, x):
        if np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            return NotImplemented

    def __matmul__(self, x):
        if np.isscalar(x):
            raise ValueError("Scalar not allowed, use * instead")
        return self.__mul__(x)

    def __rmatmul__(self, x):
        if np.isscalar(x):
            raise ValueError("Scalar not allowed, use * instead")
        return self.__rmul__(x)

    def __pow__(self, p):
        return _PowerLinearOperator(self, p)

    def __add__(self, x):
        return _SumLinearOperator(self, x)

    def __neg__(self):
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, x):
        return self.__add__(-x)

    def _adjoint(self):
        return _AdjointLinearOperator(self)

    def _transpose(self):
        return _TransposedLinearOperator(self)

    def conj(self):
        return _ConjLinearOperator(self)


class _AdjointLinearOperator(MPILinearOperator):
    """Adjoint of MPI Linear Operator"""

    def __init__(self, A: MPILinearOperator):
        self.A = A
        self.args = (A,)
        super().__init__(shape=(A.shape[1], A.shape[0]), dtype=A.dtype,
                         base_comm=MPI.COMM_WORLD)

    def _matvec(self, x: DistributedArray):
        return self.A.rmatvec(x)

    def _rmatvec(self, x: DistributedArray):
        return self.A.matvec(x)


class _TransposedLinearOperator(MPILinearOperator):
    """Transposition of MPI Linear Operator"""

    def __init__(self, A: MPILinearOperator):
        self.A = A
        self.args = (A,)
        super().__init__(shape=(A.shape[1], A.shape[0]), dtype=A.dtype,
                         base_comm=MPI.COMM_WORLD)

    def _matvec(self, x: DistributedArray):
        x[:] = np.conj(x.local_array)
        y = self.A.rmatvec(x)
        y[:] = np.conj(y.local_array)
        return y

    def _rmatvec(self, x: DistributedArray):
        x[:] = np.conj(x.local_array)
        y = self.A.matvec(x)
        y[:] = np.conj(y.local_array)
        return y


class _ProductLinearOperator(MPILinearOperator):
    """Product of MPI LinearOperators
    """
    def __init__(self, A: MPILinearOperator, B: MPILinearOperator):
        if not isinstance(A, MPILinearOperator) or not isinstance(B, MPILinearOperator):
            raise ValueError('both operands have to be a LinearOperator')
        if A.shape[1] != B.shape[0]:
            raise ValueError('cannot multiply %r and %r: shape mismatch' % (A, B))
        self.args = (A, B)
        super().__init__(shape=(A.shape[0], B.shape[1]), dtype=_get_dtype([A, B]),
                         base_comm=MPI.COMM_WORLD)

    def _matvec(self, x: DistributedArray):
        return self.args[0].matvec(self.args[1].matvec(x))

    def _rmatvec(self, x: DistributedArray):
        return self.args[1].rmatvec(self.args[0].rmatvec(x))

    def _adjoint(self):
        A, B = self.args
        return B.H * A.H


class _ScaledLinearOperator(MPILinearOperator):
    """Scaled MPI Linear Operator
    """

    def __init__(self, A: MPILinearOperator, alpha):
        if not isinstance(A, MPILinearOperator):
            raise ValueError('MPILinearOperator expected as A')
        if not np.isscalar(alpha):
            raise ValueError('scalar expected as alpha')
        self.args = (A, alpha)
        super().__init__(shape=A.shape, dtype=_get_dtype([A], [type(alpha)]),
                         base_comm=MPI.COMM_WORLD)

    def _matvec(self, x: DistributedArray):
        y = self.args[0].matvec(x)
        if y is not None:
            y[:] *= self.args[1]
        return y

    def _rmatvec(self, x: DistributedArray):
        y = self.args[0].rmatvec(x)
        if y is not None:
            y[:] *= np.conj(self.args[1])
        return y

    def _adjoint(self):
        A, alpha = self.args
        return A.H * np.conj(alpha)


class _SumLinearOperator(MPILinearOperator):
    """Sum of MPI LinearOperators
    """

    def __init__(self, A: MPILinearOperator, B: MPILinearOperator):
        if not isinstance(A, MPILinearOperator) or not isinstance(B, MPILinearOperator):
            raise ValueError('both operands have to be a MPILinearOperator')
        # Make sure it works with different kinds
        if A.shape != B.shape:
            raise ValueError("cannot add %r and %r: shape mismatch" % (A, B))
        self.args = (A, B)
        super().__init__(shape=A.shape, dtype=A.dtype, base_comm=MPI.COMM_WORLD)

    def _matvec(self, x: DistributedArray):
        arr1 = self.args[0].matvec(x)
        arr2 = self.args[1].matvec(x)
        return arr1 + arr2

    def _rmatvec(self, x):
        arr1 = self.args[0].rmatvec(x)
        arr2 = self.args[1].rmatvec(x)
        return arr1 + arr2

    def _adjoint(self):
        A, B = self.args
        return A.H + B.H


class _PowerLinearOperator(MPILinearOperator):
    """Power of MPI Linear Operator
    """

    def __init__(self, A: MPILinearOperator, p: int) -> None:
        if not isinstance(A, MPILinearOperator):
            raise ValueError("LinearOperator expected as A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("square LinearOperator expected, got %r" % A)
        if not isintlike(p) or p < 0:
            raise ValueError("non-negative integer expected as p")

        super(_PowerLinearOperator, self).__init__(shape=A.shape, dtype=A.dtype, base_comm=A.base_comm)
        self.args = (A, p)

    def _power(self, fun: Callable, x: DistributedArray):
        res = DistributedArray(global_shape=x.global_shape, dtype=x.dtype)
        res[:] = x.local_array
        for _ in range(self.args[1]):
            res[:] = fun(res).local_array
        return res

    def _matvec(self, x: DistributedArray):
        return self._power(self.args[0].matvec, x)

    def _rmatvec(self, x: DistributedArray):
        return self._power(self.args[0].rmatvec, x)


class _ConjLinearOperator(MPILinearOperator):
    """Complex conjugate MPI Linear Operator
    """

    def __init__(self, A: MPILinearOperator):
        if not isinstance(A, MPILinearOperator):
            raise TypeError('A must be a MPILinearOperator')
        self.A = A
        super().__init__(shape=A.shape, dtype=A.dtype, base_comm=MPI.COMM_WORLD)

    def _matvec(self, x: DistributedArray):
        x[:] = x.local_array.conj()
        y = self.A.matvec(x)
        if y is not None:
            y[:] = y.local_array.conj()
        return y

    def _rmatvec(self, x: DistributedArray):
        x[:] = x.local_array.conj()
        y = self.A.rmatvec(x)
        if y is not None:
            y[:] = y.local_array.conj()
        return y

    def _adjoint(self):
        return _ConjLinearOperator(self.A.H)


def asmpilinearoperator(Op):
    """Return Op as a MPI LinearOperator.

    Converts a :class:`pylops.LinearOperator` to a :class:`pylops_mpi.MPILinearOperator`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        PyLops LinearOperator

    Returns
    -------
    Op : :obj:`pylops_mpi.MPILinearOperator`
        Operator of type :obj:`pylops_mpi.MPILinearOperator`

    """
    if isinstance(Op, MPILinearOperator):
        return Op
    else:
        return MPILinearOperator(shape=Op.shape, dtype=Op.dtype, Op=Op,
                                 base_comm=MPI.COMM_WORLD)
