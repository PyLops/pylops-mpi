from typing import Optional, Tuple
import numpy as np
import time

from pylops.optimization.basesolver import Solver
from pylops.utils import NDArray

from pylops_mpi import DistributedArray


class CGLS(Solver):
    r"""Conjugate gradient least squares

    Solve an overdetermined system of equations given a MPILinearOperator ``Op``
    and distributed data ``y`` using conjugate gradient iterations.

    Parameters
    ----------
    Op : :obj:`pylops_mpi.MPILinearOperator`
        Operator to invert of size :math:`[N \times M]`

    Notes
    -----
    Minimize the following functional using conjugate gradient iterations:

    .. math::
        J = || \mathbf{y} -  \mathbf{Op}\,\mathbf{x} ||_2^2 +
        \epsilon^2 || \mathbf{x} ||_2^2

    where :math:`\epsilon` is the damping coefficient.

    """

    def _print_setup(self, xcomplex: bool = False) -> None:
        self._print_solver(nbar=65)

        if self.niter is not None:
            strpar = (
                f"damp = {self.damp:10e}\ttol = {self.tol:10e}\tniter = {self.niter}"
            )
        else:
            strpar = f"damp = {self.damp:10e}\ttol = {self.tol:10e}\t"
        print(strpar)
        print("-" * 65 + "\n")
        if not xcomplex:
            head1 = "    Itn          x[0]              r1norm         r2norm"
        else:
            head1 = "    Itn             x[0]             r1norm         r2norm"
        print(head1)

    def _print_step(self, x: DistributedArray) -> None:
        strx = f"{x[0]:1.2e}   " if np.iscomplexobj(x) else f"{x[0]:11.4e}        "
        msg = (
            f"{self.iiter:6g}       "
            + strx
            + f"{self.cost[self.iiter]:11.4e}    {self.cost1[self.iiter]:11.4e}"
        )
        print(msg)

    def setup(self,
              y: DistributedArray,
              x0: Optional[DistributedArray] = None,
              niter: Optional[int] = None,
              damp: float = 0.0,
              tol: float = 1e-4,
              show: bool = False,
              ) -> DistributedArray:
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`pylops_mpi.DistributedArray`
            Data of size :math:`[N \times 1]`
        x0 : :obj:`pylops_mpi.DistributedArray`, optional
            Initial guess  of size (M,). If ``None``, Defaults to a
            zero vector
        niter : :obj:`int`, optional
            Number of iterations (default to ``None`` in case a user wants to
            manually step over the solver)
        damp : :obj:`float`, optional
            Damping coefficient
        tol : :obj:`float`, optional
            Tolerance on residual norm
        show : :obj:`bool`, optional
            Display setup log

        Returns
        -------
        x : :obj:`pylops_mpi.DistributedArray`
            Initial guess of size (N,).

        """
        self.y = y
        self.damp = damp ** 2
        self.tol = tol
        self.niter = niter

        # initialize solver
        if x0 is None:
            x = DistributedArray(global_shape=self.Op.shape[1], dtype=y.dtype)
            x[:] = 0
            self.s = y.copy()
            r = self.Op.rmatvec(self.s)
        else:
            x = x0.copy()
            self.s = self.y - self.Op.matvec(x)
            damped_x = DistributedArray(global_shape=x.global_shape, dtype=x.dtype)
            damped_x[:] = damp * x.local_array
            r = self.Op.rmatvec(self.s) - damped_x
        self.c = r.copy()
        self.q = self.Op.matvec(self.c)
        self.kold = np.abs(r.dot(r.conj()))

        # create variables to track the residual norm and iterations
        self.cost = []
        self.cost1 = []
        self.cost.append(float(self.s.norm()))
        self.cost1.append(np.sqrt(float(self.cost[0] ** 2 + damp * np.abs(x.dot(x.conj())))))
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(np.issubdtype(x.dtype, np.complex128))
        return x

    def step(self, x: DistributedArray, show: bool = False) -> DistributedArray:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`pylops_mpi.DistributedArray`
            Current model vector to be updated by a step of CG
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`pylops_mpi.DistributedArray`
            Updated model vector

        """

        a = self.kold / (self.q.dot(self.q.conj()) + self.damp * self.c.dot(self.c.conj()))
        x[:] = x.local_array + a * self.c.local_array
        self.s[:] = self.s.local_array - a * self.q.local_array
        damped_x = DistributedArray(global_shape=x.global_shape, dtype=x.dtype)
        damped_x[:] = self.damp * x.local_array
        r = self.Op.rmatvec(self.s) - damped_x
        k = np.abs(r.dot(r.conj()))
        b = k / self.kold
        self.c[:] = r.local_array + b * self.c.local_array
        self.q = self.Op.matvec(self.c)
        self.kold = k
        self.iiter += 1
        self.cost.append(float(self.s.norm()))
        self.cost1.append(np.sqrt(float(self.cost[self.iiter] ** 2 + self.damp * np.abs(x.dot(x.conj())))))
        if show:
            self._print_step(x)
        return x

    def run(self,
            x: DistributedArray,
            niter: Optional[int] = None,
            show: bool = False,
            itershow: Tuple[int, int, int] = (10, 10, 10), ) -> DistributedArray:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`pylops_mpi.DistributedArray`
            Current model vector to be updated by multiple steps of CGLS
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display iterations log
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`pylops_mpi.DistributedArray`
            Estimated model of size (M, ).

        """
        niter = self.niter if niter is None else niter
        if niter is None:
            raise ValueError("niter must not be None")
        while self.iiter < niter and self.kold > self.tol:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x = self.step(x, showstep)
            self.callback(x)
        return x

    def callback(self, x: DistributedArray, **kwargs) -> None:
        """Callback routine

        This routine must be passed by the user. Its function signature must contain
        a single input that contains the current solution. It will be invoked at each step
        of the `solve` method.

        Parameters
        ----------
        x::obj: `pylops_mpi.DistributedArray`
            Current solution

        """

        pass

    def finalize(self, show: bool = False, **kwargs) -> None:
        r"""Finalize solver

        Parameters
        ----------
        show : :obj:`bool`, optional
            Display finalize log

        """

        self.tend = time.time()
        self.telapsed = self.tend - self.tstart
        # reason for termination
        self.istop = 1 if self.kold < self.tol else 2
        self.r1norm = self.kold
        self.r2norm = self.cost1[self.iiter]
        if show:
            self._print_finalize(nbar=65)
        self.cost = np.array(self.cost)

    def solve(self,
              y: DistributedArray,
              x0: Optional[DistributedArray] = None,
              niter: int = 10,
              damp: float = 0.0,
              tol: float = 1e-4,
              show: bool = False,
              itershow: Tuple[int, int, int] = (10, 10, 10),
              ) -> Tuple[DistributedArray, int, int, float, float, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        y : :obj:`pylops_mpi.DistributedArray`
            Data of size (N, )
        x0 : :obj:`pylops_mpi.DistributedArray`
            Initial guess  of size (M, ). If ``None``, initialize
            internally as zero vector
        niter : :obj:`int`, optional
            Number of iterations (default to ``None`` in case a user wants to
            manually step over the solver)
        damp : :obj:`float`, optional
            Damping coefficient
        tol : :obj:`float`, optional
            Tolerance on residual norm
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`pylops_mpi.DistributedArray`
            Estimated model of size (M, ).
        istop : :obj:`int`
            Gives the reason for termination

            ``1`` means :math:`\mathbf{x}` is an approximate solution to
            :math:`\mathbf{y} = \mathbf{Op}\,\mathbf{x}`

            ``2`` means :math:`\mathbf{x}` approximately solves the least-squares
            problem
        iit : :obj:`int`
            Iteration number upon termination
        r1norm : :obj:`float`
            :math:`||\mathbf{r}||_2`, where
            :math:`\mathbf{r} = \mathbf{y} - \mathbf{Op}\,\mathbf{x}`
        r2norm : :obj:`float`
            :math:`\sqrt{\mathbf{r}^T\mathbf{r}  +
            \epsilon^2 \mathbf{x}^T\mathbf{x}}`.
            Equal to ``r1norm`` if :math:`\epsilon=0`
        cost : :obj:`numpy.ndarray`, optional
            History of r1norm through iterations

        """

        x = self.setup(y=y, x0=x0, niter=niter, damp=damp, tol=tol, show=show)
        x = self.run(x, niter, show=show, itershow=itershow)
        self.finalize(show)
        return x, self.istop, self.iiter, self.r1norm, self.r2norm, self.cost
