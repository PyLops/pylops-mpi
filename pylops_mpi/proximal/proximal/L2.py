from math import sqrt
from mpi4py import MPI
from typing import TYPE_CHECKING, Any, Callable

from pylops.basicoperators import Identity
from pyproximal.ProxOperator import _check_tau

from pylops_mpi import DistributedArray, StackedDistributedArray
from pylops_mpi.basicoperators import MPIBlockDiag, MPIStackedVStack
from pylops_mpi.optimization.basic import cg, cgls
from pylops_mpi.proximal import MPIProxOperator

if TYPE_CHECKING:
    from pylops_mpi import MPILinearOperator


class MPIL2(MPIProxOperator):
    """L2 Norm proximal operator.

    Implement a distributed version of the L2 norm proximal operator.

    Parameters
    ----------
    Op : :obj:`pylops_mpi.MPILinearOperator`, optional
        MPI-enabled PyLops Linear Operator
    b : :obj:`pylops_mpi.DistributedArray`, optional
        Data vector
    q : :obj:`pylops_mpi.DistributedArray`, optional
        Dot vector
    sigma : :obj:`int`, optional
        Multiplicative coefficient of L2 norm
    alpha : :obj:`float`, optional
        Multiplicative coefficient of dot product
    qgrad : :obj:`bool`, optional
        Add q term to gradient (``True``) or not (``False``)
    niter : :obj:`int` or :obj:`func`, optional
        Number of iterations of iterative scheme used to compute the proximal.
        This can be a constant number or a function that is called passing a
        counter which keeps track of how many times the ``prox`` method has
        been invoked before and returns the ``niter`` to be used.
    x0 : :obj:`pylops_mpi.DistributedArray`, optional
        Initial vector. If ``Op`` is not None, this must be passed.
    warm : :obj:`bool`, optional
        Warm start (``True``) or not (``False``). Uses estimate from previous
        call of ``prox`` method.
    solver : :obj:`str`, optional
        .. versionadded:: 0.11.0

        Name of solver to use with non-explicit operators:

        - ``cg`` to use :py:func:`pylops_mpi.optimization.basic.cg` on the
          normal equations;
        - ``cgls`` to use :py:func:`pylops.optimization.basic.cgls` on the
          regularized system of equations;
    **kwargs_solver : :obj:`dict`, optional
        Dictionary containing extra arguments for the solver selected
        via the ``solver`` parameter.

    """

    def __init__(
        self,
        Op: "MPILinearOperator" = None,
        b: DistributedArray | None = None,
        q: DistributedArray | None = None,
        sigma: float = 1.0,
        alpha: float = 1.0,
        qgrad: bool = True,
        niter: int | Callable[[int], int] = 10,
        x0: DistributedArray | None = None,
        warm: bool = True,
        solver: str | None = "cgls",
        kwargs_solver: dict[str, Any] | None = None,
    ) -> None:
        if Op is not None and x0 is None:
            raise ValueError("x0 must be passed when Op is not None")
        self.Op = Op
        self.hasgrad = True

        self.b = b
        self.q = q
        self.sigma = sigma
        self.alpha = alpha
        self.qgrad = qgrad
        self.niter = niter
        self.x0 = x0
        self.warm = warm
        self.solver = solver
        self.count = 0
        self.kwargs_solver = {} if kwargs_solver is None else kwargs_solver

        # define whether the normal equations or the regularized system
        # of equations are solved
        if self.solver == "cg":
            self.normaleqs = True
        elif self.solver == "cgls":
            self.normaleqs = False
        else:
            msg = (
                f"Provided solver={self.solver}. "
                "Available options are 'cg' or 'cgls'."
            )
            raise ValueError(msg)
        
        # create data term
        if (
            self.Op is not None
            and self.b is not None
            and self.normaleqs
        ):
            self.OpTb = self.sigma * self.Op.H @ self.b

    def __call__(self, x: DistributedArray) -> DistributedArray:
        if self.Op is not None and self.b is not None:
            f = (self.sigma / 2.0) * ((self.Op * x - self.b).norm() ** 2)
        elif self.b is not None:
            f = (self.sigma / 2.0) * ((x - self.b).norm() ** 2)
        else:
            f = (self.sigma / 2.0) * (x.norm() ** 2)
        if self.q is not None:
            f += self.alpha * self.q.dot(x)
        return float(f)
    

    def _increment_count(func: Callable[..., Any]) -> Callable[..., Any]:
        """Increment counter"""

        def wrapped(self, *args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return func(self, *args, **kwargs)

        return wrapped

    @_increment_count
    @_check_tau
    def prox(self, x: DistributedArray, tau: float, **kwargs: Any) -> DistributedArray:
        """Proximal operator applied to a vector
        """
        # define current number of iterations
        if isinstance(self.niter, int):
            niter = self.niter
        else:
            niter = self.niter(self.count)

        # solve proximal optimization
        if self.Op is not None and self.b is not None:
            if self.normaleqs:
                y = x + tau * self.OpTb
                if self.q is not None:
                    y -= tau * self.alpha * self.q
            if self.normaleqs:
                Op1 = MPIBlockDiag([Identity(x.local_shape, dtype=self.Op.dtype, )]) + float(
                    tau * self.sigma
                ) * (self.Op.H * self.Op)
                x = cg(Op1, y, niter=niter, x0=self.x0, **self.kwargs_solver)[0]
            else:
                y = x
                if self.q is not None:
                    y -= tau * self.alpha * self.q
                
                Opreg = MPIStackedVStack([
                    sqrt(tau * self.sigma) * self.Op,
                    MPIBlockDiag([Identity(x.local_shape, dtype=self.Op.dtype, ),])])
                breg = StackedDistributedArray([sqrt(tau * self.sigma) * self.b, y])
                x = cgls(Opreg, breg, x0=self.x0, niter=niter, **self.kwargs_solver)[0]
            if self.warm:
                self.x0 = x
        elif self.b is not None:
            num = x + tau * self.sigma * self.b
            if self.q is not None:
                num -= tau * self.alpha * self.q
            x = (1. / (1.0 + tau * self.sigma)) * num
        else:
            num = x
            if self.q is not None:
                num -= tau * self.alpha * self.q
            x = (1.0 / (1.0 + tau * self.sigma)) * num
        return x

    def grad(self, x: DistributedArray) -> DistributedArray:
        if self.Op is not None and self.b is not None:
            g = self.sigma * self.Op.H @ (self.Op @ x - self.b)
        elif self.b is not None:
            g = self.sigma * (x - self.b)
        else:
            g = self.sigma * x
        if self.q is not None and self.qgrad:
            g += self.alpha * self.q
        return g
