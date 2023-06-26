import numpy as np
import scipy as sp
from mpi4py import MPI
from typing import Callable

# need to check scipy version since the interface submodule changed into
# _interface from scipy>=1.8.0
sp_version = sp.__version__.split(".")
if int(sp_version[0]) <= 1 and int(sp_version[1]) < 8:
    from scipy.sparse.sputils import isintlike
    from scipy.sparse.linalg.interface import _get_dtype
else:
    from scipy.sparse._sputils import isintlike
    from scipy.sparse.linalg._interface import _get_dtype

from pylops import LinearOperator
from pylops.utils import DTypeLike, ShapeLike

from pylops_mpi import DistributedArray


class MPILinearOperator(LinearOperator):
    def __init__(self, shape: ShapeLike, dtype: DTypeLike, Op: LinearOperator = None, kind='all',
                 explicit=False, base_comm: MPI.Comm = MPI.COMM_WORLD):
        self.Op = Op
        self.kind = kind
        self.dtype = dtype
        self.shape = shape
        self.base_comm = base_comm
        self.size = self.base_comm.Get_size()
        self.rank = self.base_comm.Get_rank()
        super().__init__(Op=Op, explicit=explicit)

    def matvec(self, x):
        if self.kind in ("all", "force", "mix"):
            if self.Op:
                op_shapes = self.base_comm.allgather((self.Op.shape[0], ))
                y = DistributedArray(global_shape=np.sum(op_shapes), local_shapes=op_shapes)
                x = DistributedArray.to_dist(x=x)
                y[:] = self.Op._matvec(x.local_array)
            else:
                y = self._matvec(x)
        elif self.kind == 'master':
            if self.Op:
                y = self.Op._matvec(x) if self.rank == 0 else None
            else:
                y = self._matvec(x) if self.rank == 0 else None
        else:
            raise KeyError('kind must be all, master, mix or force')
        if isinstance(y, DistributedArray):
            y = y.asarray()
        return y

    def rmatvec(self, x):
        if self.kind in ('all', 'mix', 'force'):
            if self.Op:
                op_shapes = self.base_comm.allgather((self.Op.shape[1],))
                y = DistributedArray(global_shape=np.sum(op_shapes), local_shapes=op_shapes)
                x = DistributedArray.to_dist(x=x)
                y[:] = self.Op._rmatvec(x.local_array)
            else:
                y = self._rmatvec(x)
        elif self.kind == 'master':
            if self.Op:
                y = self.Op._rmatvec(x) if self.rank == 0 else None
            else:
                y = self._rmatvec(x) if self.rank == 0 else None
        else:
            raise KeyError('kind must be all, master, mix or force')
        if isinstance(y, DistributedArray):
            y = y.asarray()
        return y

    def dot(self, x):
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
    """Adjoint of arbitrary MPI Linear Operator"""

    def __init__(self, A: MPILinearOperator):
        self.A = A
        self.kind = A.kind
        self.args = (A,)
        super().__init__(shape=(A.shape[1], A.shape[0]), dtype=A.dtype,
                         explicit=A.explicit, base_comm=MPI.COMM_WORLD,
                         kind=self.kind)

    def _matvec(self, x):
        return self.A.rmatvec(x)

    def _rmatvec(self, x):
        return self.A.matvec(x)


class _TransposedLinearOperator(MPILinearOperator):
    """Transposition of arbitrary MPI Linear Operator"""

    def __init__(self, A: MPILinearOperator):
        self.A = A
        self.kind = A.kind
        self.args = (A,)
        super().__init__(shape=(A.shape[1], A.shape[0]), dtype=A.dtype,
                         explicit=A.explicit, base_comm=MPI.COMM_WORLD,
                         kind=self.kind)

    def _matvec(self, x):
        arr = DistributedArray.to_dist(x=self.A.rmatvec(np.conj(x)))
        arr[:] = np.conj(arr.local_array)
        return arr

    def _rmatvec(self, x):
        arr = DistributedArray.to_dist(x=self.A.matvec(np.conj(x)))
        arr[:] = np.conj(arr.local_array)
        return arr


class _ProductLinearOperator(MPILinearOperator):
    """Product MPI Linear Operator
    """
    def __init__(self, A: MPILinearOperator, B: MPILinearOperator):
        if not isinstance(A, MPILinearOperator) or not isinstance(B, MPILinearOperator):
            raise ValueError('both operands have to be a LinearOperator')
        # Make sure it works with different kinds
        shape_A, shape_B = (A.shape, B.shape)
        if A.kind == "all":
            shape_A = tuple(np.sum(A.base_comm.allgather(A.shape), axis=0))
        if B.kind == "all":
            shape_B = tuple(np.sum(B.base_comm.allgather(B.shape), axis=0))
        if shape_A[1] != shape_B[0]:
            raise ValueError('cannot multiply %r and %r: shape mismatch' % (A, B))
        self.args = (A, B)
        self.kind = 'mix'
        super().__init__(shape=(A.shape[0], B.shape[1]), dtype=_get_dtype([A, B]),
                         explicit=A.explicit, base_comm=MPI.COMM_WORLD, kind=self.kind)

    def _matvec(self, x):
        return self.args[0].matvec(self.args[1].matvec(x))

    def _rmatvec(self, x):
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
        self.kind = A.kind
        super().__init__(shape=A.shape, dtype=_get_dtype([A], [type(alpha)]),
                         explicit=A.explicit, base_comm=MPI.COMM_WORLD, kind=self.kind)

    def _matvec(self, x):
        y = DistributedArray.to_dist(self.args[0].matvec(x))
        if y is not None:
            y[:] *= self.args[1]
        return y

    def _rmatvec(self, x):
        y = DistributedArray.to_dist(x=self.args[0].rmatvec(x))
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
        shape_A, shape_B = (A.shape, B.shape)
        if A.kind == "all":
            shape_A = tuple(np.sum(A.base_comm.allgather(A.shape), axis=0))
        if B.kind == "all":
            shape_B = tuple(np.sum(B.base_comm.allgather(B.shape), axis=0))
        if shape_A != shape_B:
            raise ValueError("cannot add %r and %r: shape mismatch" % (A, B))
        self.args = (A, B)
        self.kind = 'mix'
        super().__init__(shape=A.shape, dtype=A.dtype, explicit=A.explicit,
                         base_comm=MPI.COMM_WORLD, kind=self.kind)

    def _matvec(self, x):
        arr1 = DistributedArray.to_dist(x=self.args[0].matvec(x))
        arr2 = DistributedArray.to_dist(x=self.args[1].matvec(x))
        return arr1 + arr2

    def _rmatvec(self, x):
        arr1 = DistributedArray.to_dist(x=self.args[0].rmatvec(x))
        arr2 = DistributedArray.to_dist(x=self.args[1].rmatvec(x))
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

        super(_PowerLinearOperator, self).__init__(shape=A.shape, dtype=A.dtype, kind=A.kind,
                                                   explicit=A.explicit, base_comm=A.base_comm)
        self.args = (A, p)

    def _power(self, fun: Callable, x):
        res = np.array(x, copy=True)
        for _ in range(self.args[1]):
            res = fun(res)
        return res

    def _matvec(self, x):
        return self._power(self.args[0].matvec, x)

    def _rmatvec(self, x):
        return self._power(self.args[0].rmatvec, x)


class _ConjLinearOperator(MPILinearOperator):
    """Complex conjugate linear operator
    """

    def __init__(self, A: MPILinearOperator):
        if not isinstance(A, MPILinearOperator):
            raise TypeError('A must be a MPILinearOperator')
        self.kind = A.kind
        self.A = A
        super().__init__(shape=A.shape, dtype=A.dtype,
                         explicit=A.explicit, base_comm=MPI.COMM_WORLD,
                         kind=self.kind)

    def _matvec(self, x):
        y = DistributedArray.to_dist(x=self.A.matvec(x.conj()))
        if y is not None:
            y[:] = y.local_array.conj()
        return y

    def _rmatvec(self, x):
        y = DistributedArray.to_dist(x=self.A.rmatvec(x.conj()))
        if y is not None:
            y[:] = y.local_array.conj()
        return y

    def _adjoint(self):
        return _ConjLinearOperator(self.A.H)


def asmpilinearoperator(Op, kind: str = "all"):
    if isinstance(Op, MPILinearOperator):
        return Op
    else:
        return MPILinearOperator(shape=Op.shape, dtype=Op.dtype, Op=Op,
                                 explicit=Op.explicit, base_comm=MPI.COMM_WORLD,
                                 kind=kind)
