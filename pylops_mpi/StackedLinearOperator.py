from typing import Optional, Union, Callable
from abc import abstractmethod, ABC
import numpy as np
from mpi4py import MPI
from pylops.utils import ShapeLike, DTypeLike

from scipy.sparse._sputils import isintlike
from scipy.sparse.linalg._interface import _get_dtype

from pylops_mpi.DistributedArray import DistributedArray, StackedDistributedArray


class MPIStackedLinearOperator(ABC):
    """Common interface for performing matrix-vector products in distributed fashion
    for StackedLinearOperators.

    This class provides methods to perform matrix-vector product and adjoint matrix-vector
    products on a stack of MPILinearOperator objects.

    .. note:: End users of pylops-mpi should not use this class directly but simply
    use operators that are already implemented. This class is meant for
    developers only, it has to be used as the parent class of any new operator
    developed within pylops-mpi.

    Parameters
    ----------
    shape : :obj:`tuple(int, int)`, optional
        Shape of the MPIStackedLinearOperator. Defaults to ``None``.
    dtype : :obj:`str`, optional
        Type of elements in input array. Defaults to ``None``.
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        MPI Base Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
    """

    def __init__(self, shape: Optional[ShapeLike] = None,
                 dtype: Optional[DTypeLike] = None,
                 base_comm: MPI.Comm = MPI.COMM_WORLD):
        if shape:
            self.shape = shape
        if dtype:
            self.dtype = dtype
        # For MPI
        self.base_comm = base_comm
        self.size = self.base_comm.Get_size()
        self.rank = self.base_comm.Get_rank()

    def matvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        M, N = self.shape
        if isinstance(x, StackedDistributedArray):
            stacked_shape = (np.sum([a.global_shape for a in x.distarrays]), )
            if stacked_shape != (N, ):
                raise ValueError("dimension mismatch")
        if isinstance(x, DistributedArray) and x.global_shape != (N,):
            raise ValueError("dimension mismatch")
        return self._matvec(x)

    @abstractmethod
    def _matvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        pass

    def rmatvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        M, N = self.shape
        if isinstance(x, StackedDistributedArray):
            stacked_shape = (np.sum([a.global_shape for a in x.distarrays]), )
            if stacked_shape != (M, ):
                raise ValueError("dimension mismatch")
        if isinstance(x, DistributedArray) and x.global_shape != (M,):
            raise ValueError("dimension mismatch")
        return self._rmatvec(x)

    @abstractmethod
    def _rmatvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        pass

    def dot(self, x):
        """Matrix Vector Multiplication

        Parameters
        ----------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray` or
            :obj:`pylops_mpi.StackedMPILinearOperator
            StackedDistributedArray, DistributedArray or StackedMPILinearOperator.

        Returns
        -------
        y : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray` or
            :obj:`pylops_mpi.StackedMPILinearOperator
            StackedDistributedArray, DistributedArray or a StackedMPILinearOperator.

        """
        if isinstance(x, MPIStackedLinearOperator):
            return _ProductStackedLinearOperator(self, x)
        elif np.isscalar(x):
            return _ScaledStackedLinearOperator(self, x)
        else:
            if x is None or (isinstance(x, DistributedArray) and x.ndim == 1):
                return self.matvec(x)
            elif isinstance(x, StackedDistributedArray):
                ndims = np.unique([dis.ndim for dis in x.distarrays])
                if len(ndims) == 1 and ndims[0] == 1:
                    return self.matvec(x)
            else:
                raise ValueError('expected 1-d DistributedArray or StackedDistributedArray')

    def adjoint(self):
        """Adjoint MPIStackedLinearOperator

        Returns
        -------
        op : :obj:`pylops_mpi.MPIStackedLinearOperator`
            Adjoint of Operator

        """
        return self._adjoint()

    H = property(adjoint)

    def transpose(self):
        """Transposition of MPIStackedLinearOperator

        Returns
        -------
        op : :obj:`pylops_mpi.MPIStackedLinearOperator`
            Transpose MPIStackedLinearOperator

        """
        return self._transpose()

    T = property(transpose)

    def __mul__(self, x):
        return self.dot(x)

    def __rmul__(self, x):
        if np.isscalar(x):
            return _ScaledStackedLinearOperator(self, x)
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
        return _SumStackedLinearOperator(self, x)

    def __neg__(self):
        return _ScaledStackedLinearOperator(self, -1)

    def __sub__(self, x):
        return self.__add__(-x)

    def _adjoint(self):
        return _AdjointStackedLinearOperator(self)

    def _transpose(self):
        return _TransposedStackedLinearOperator(self)

    def conj(self):
        """Complex conjugate operator

        Returns
        -------
        conjop : :obj:`pylops_mpi.MPIStackedLinearOperator`
            Complex conjugate operator

        """
        return _ConjLinearOperator(self)

    def __repr__(self):
        M, N = self.shape
        if self.dtype is None:
            dt = "unspecified dtype"
        else:
            dt = f"dtype={self.dtype}"
        return f"<{M}x{N} {self.__class__.__name__} with {dt}>"


class _AdjointStackedLinearOperator(MPIStackedLinearOperator):
    """Adjoint of MPIStackedLinearOperator"""

    def __init__(self, A: MPIStackedLinearOperator):
        self.A = A
        self.args = (A,)
        super().__init__(shape=(A.shape[1], A.shape[0]), dtype=A.dtype,
                         base_comm=MPI.COMM_WORLD)

    def _matvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        return self.A.rmatvec(x)

    def _rmatvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        return self.A.matvec(x)


class _TransposedStackedLinearOperator(MPIStackedLinearOperator):
    """Transpose of MPIStackedLinearOperator"""

    def __init__(self, A: MPIStackedLinearOperator):
        self.A = A
        self.args = (A,)
        super().__init__(shape=(A.shape[1], A.shape[0]), dtype=A.dtype,
                         base_comm=MPI.COMM_WORLD)

    def _matvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        x = x.conj()
        y = self.A.rmatvec(x)
        y = y.conj()
        return y

    def _rmatvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        x = x.conj()
        y = self.A.matvec(x)
        y = y.conj()
        return y


class _ProductStackedLinearOperator(MPIStackedLinearOperator):
    """Product of MPI Stacked Linear Operators"""
    def __init__(self, A: MPIStackedLinearOperator, B: MPIStackedLinearOperator):
        from pylops_mpi.basicoperators.VStack import MPIStackedVStack
        from pylops_mpi.basicoperators.BlockDiag import MPIStackedBlockDiag
        if not isinstance(A, MPIStackedLinearOperator) or not isinstance(B, MPIStackedLinearOperator):
            raise ValueError('both operands have to be a MPIStackedLinearOperator')
        if isinstance(A, MPIStackedVStack) and isinstance(B, MPIStackedVStack):
            raise ValueError('both operands cannot be MPIStackedVStack')
        if isinstance(A, MPIStackedBlockDiag) and isinstance(B, MPIStackedBlockDiag) and len(A.ops) != len(B.ops):
            raise ValueError(f'both MPIStackedBlockDiag cannot have different number of ops, {A.ops} != {B.ops}')
        if A.shape[1] != B.shape[0]:
            raise ValueError('cannot multiply %r and %r: shape mismatch' % (A, B))
        self.args = (A, B)
        super().__init__(shape=(A.shape[0], B.shape[1]), dtype=_get_dtype([A, B]),
                         base_comm=MPI.COMM_WORLD)

    def _matvec(self, x: Union[StackedDistributedArray, DistributedArray]) -> Union[StackedDistributedArray, DistributedArray]:
        return self.args[0].matvec(self.args[1].matvec(x))

    def _rmatvec(self, x: Union[StackedDistributedArray, DistributedArray]) -> Union[StackedDistributedArray, DistributedArray]:
        return self.args[1].rmatvec(self.args[0].rmatvec(x))

    def _adjoint(self) -> MPIStackedLinearOperator:
        A, B = self.args
        return B.H * A.H


class _ScaledStackedLinearOperator(MPIStackedLinearOperator):
    """Scaled MPI StackedLinearOperator
    """

    def __init__(self, A: MPIStackedLinearOperator, alpha):
        if not isinstance(A, MPIStackedLinearOperator):
            raise ValueError('MPILinearOperator expected as A')
        if not np.isscalar(alpha):
            raise ValueError('scalar expected as alpha')
        self.args = (A, alpha)
        super().__init__(shape=A.shape, dtype=_get_dtype([A], [type(alpha)]),
                         base_comm=MPI.COMM_WORLD)

    def _matvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        y = self.args[0].matvec(x)
        if y is not None:
            y *= self.args[1]
        return y

    def _rmatvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        y = self.args[0].rmatvec(x)
        if y is not None:
            y *= np.conj(self.args[1])
        return y

    def _adjoint(self) -> MPIStackedLinearOperator:
        A, alpha = self.args
        return A.H * np.conj(alpha)


class _SumStackedLinearOperator(MPIStackedLinearOperator):
    """Sum of MPI StackedLinearOperators
    """

    def __init__(self, A: MPIStackedLinearOperator, B: MPIStackedLinearOperator):
        if not isinstance(A, MPIStackedLinearOperator) or not isinstance(B, MPIStackedLinearOperator):
            raise ValueError('both operands have to be a MPIStackedLinearOperator')
        if type(A) != type(B):  # noqa: E721
            raise ValueError(f'both operands have to be of same type, {A} != {B}')
        if A.shape != B.shape:
            raise ValueError("cannot add %r and %r: shape mismatch" % (A, B))
        self.args = (A, B)
        super().__init__(shape=A.shape, dtype=A.dtype, base_comm=MPI.COMM_WORLD)

    def _matvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        arr1 = self.args[0].matvec(x)
        arr2 = self.args[1].matvec(x)
        return arr1 + arr2

    def _rmatvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        arr1 = self.args[0].rmatvec(x)
        arr2 = self.args[1].rmatvec(x)
        return arr1 + arr2

    def _adjoint(self) -> MPIStackedLinearOperator:
        A, B = self.args
        return A.H + B.H


class _PowerLinearOperator(MPIStackedLinearOperator):
    """Power of MPI StackedLinearOperator
    """

    def __init__(self, A: MPIStackedLinearOperator, p: int) -> None:
        if not isinstance(A, MPIStackedLinearOperator):
            raise ValueError("MPIStackedLinearOperator expected as A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("square MPIStackedLinearOperator expected, got %r" % A)
        if not isintlike(p) or p < 0:
            raise ValueError("non-negative integer expected as p")

        super(_PowerLinearOperator, self).__init__(shape=A.shape, dtype=A.dtype, base_comm=A.base_comm)
        self.args = (A, p)

    def _power(self, fun: Callable, x: Union[StackedDistributedArray, DistributedArray]) -> Union[StackedDistributedArray, DistributedArray]:
        res = x.copy()
        for _ in range(self.args[1]):
            res[:] = fun(res)[:]
        return res

    def _matvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        return self._power(self.args[0].matvec, x)

    def _rmatvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        return self._power(self.args[0].rmatvec, x)


class _ConjLinearOperator(MPIStackedLinearOperator):
    """Complex conjugate MPI StackedLinearOperator
    """

    def __init__(self, A: MPIStackedLinearOperator):
        if not isinstance(A, MPIStackedLinearOperator):
            raise TypeError('A must be a MPIStackedLinearOperator')
        self.A = A
        super().__init__(shape=A.shape, dtype=A.dtype, base_comm=MPI.COMM_WORLD)

    def _matvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        x = x.conj()
        y = self.A.matvec(x)
        if y is not None:
            y = y.conj()
        return y

    def _rmatvec(self, x: Union[DistributedArray, StackedDistributedArray]) -> Union[DistributedArray, StackedDistributedArray]:
        x = x.conj()
        y = self.A.rmatvec(x)
        if y is not None:
            y = y.conj()
        return y

    def _adjoint(self) -> MPIStackedLinearOperator:
        return _ConjLinearOperator(self.A.H)
