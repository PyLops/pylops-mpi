import sys
import time
from typing import List, Optional, Tuple, Union

import numpy as np
from pylops.optimization.basesolver import Solver
from pylops.utils import NDArray

from pylops_mpi import DistributedArray, StackedDistributedArray


class CG(Solver):
    r"""Conjugate gradient

    Solve a square system of equations given either an MPILinearOperator or an MPIStackedLinearOperator ``Op`` and
    distributed data ``y`` using conjugate gradient iterations.

    Parameters
    ----------
    Op : :obj:`pylops_mpi.MPILinearOperator` or :obj:`pylops_mpi.MPIStackedLinearOperator`
        Operator to invert of size :math:`[N \times N]`

    Notes
    -----
    Solve the :math:`\mathbf{y} = \mathbf{Op}\,\mathbf{x}` problem using conjugate gradient
    iterations.

    """

    def _print_setup(self, xcomplex: bool = False) -> None:
        self._print_solver(nbar=55)

        if self.niter is not None:
            strpar = f"tol = {self.tol:10e}\tniter = {self.niter}"
        else:
            strpar = f"tol = {self.tol:10e}"
        print(strpar)
        print("-" * 55 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]              r2norm"
        else:
            head1 = "    Itn              x[0]                  r2norm"
        print(head1)
        sys.stdout.flush()

    def _print_step(self, x: Union[DistributedArray, StackedDistributedArray]) -> None:
        if isinstance(x, StackedDistributedArray):
            x = x.distarrays[0]
        strx = (
            f"{x[0]:1.2e}        "
            if np.iscomplexobj(x.local_array)
            else f"{x[0]:11.4e}        "
        )
        msg = f"{self.iiter:6g}        " + strx + f"{self.cost[self.iiter]:11.4e}"
        print(msg)
        sys.stdout.flush()

    def memory_usage(self) -> None:
        pass

    def setup(
        self,
        y: Union[DistributedArray, StackedDistributedArray],
        x0: Union[DistributedArray, StackedDistributedArray],
        niter: Optional[int] = None,
        tol: float = 1e-4,
        show: bool = False,
    ) -> Union[DistributedArray, StackedDistributedArray]:
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Data of size (N,)
        x0 : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Initial guess of size (N,).
        niter : :obj:`int`, optional
            Number of iterations (default to ``None`` in case a user wants to manually step over the solver)
        tol : :obj:`float`, optional
            Tolerance on residual norm
        show : :obj:`bool`, optional
            Display setup log

        Returns
        -------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Initial guess of size (N,).

        """

        self.y = y
        self.niter = niter
        self.tol = tol

        x = x0.copy()
        # residual computed in-place into the output of the matvec
        self.r = self.Op.matvec(x)
        self.y.subtract(self.r, out=self.r)
        self.rank = x.rank
        self.c = self.r.copy()
        self.kold = float(np.abs(self.r.dot(self.r, vdot=True)))

        # create variables to track the residual norm and iterations
        self.cost: List = []
        self.cost.append(float(np.sqrt(self.kold)))
        self.iiter = 0

        if show and self.rank == 0:
            if isinstance(x, StackedDistributedArray):
                is_complex = any(np.iscomplexobj(x1.local_array) for x1 in x.distarrays)
                self._print_setup(is_complex)
            else:
                self._print_setup(np.iscomplexobj(x.local_array))
        return x

    def step(
        self, x: Union[DistributedArray, StackedDistributedArray], show: bool = False
    ) -> Union[DistributedArray, StackedDistributedArray]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Current model vector to be updated by a step of CG
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Updated model vector

        """
        Opc = self.Op.matvec(self.c)
        cOpc = np.abs(self.c.dot(Opc.conj()))
        a = float(self.kold / cOpc)
        # all updates are done in-place, reusing Opc as scratch space
        Opc *= a
        self.r -= Opc
        self.c.multiply(a, out=Opc)
        x += Opc
        k = float(np.abs(self.r.dot(self.r, vdot=True)))
        b = float(k / self.kold)
        self.c *= b
        self.c += self.r
        self.kold = k
        self.iiter += 1
        self.cost.append(float(np.sqrt(self.kold)))
        if show and self.rank == 0:
            self._print_step(x)
        return x

    def run(
        self,
        x: Union[DistributedArray, StackedDistributedArray],
        niter: Optional[int] = None,
        show: bool = False,
        itershow: Tuple[int, int, int] = (10, 10, 10),
    ) -> Union[DistributedArray, StackedDistributedArray]:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Current model vector to be updated by multiple steps of CG
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Estimated model of size (M,)

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

    def finalize(self, show: bool = False) -> None:
        r"""Finalize solver

        Parameters
        ----------
        show : :obj:`bool`, optional
            Display finalize log

        """

        self.tend = time.time()
        self.telapsed = self.tend - self.tstart
        self.cost = np.array(self.cost)
        if show and self.rank == 0:
            self._print_finalize(nbar=55)

    def solve(
        self,
        y: Union[DistributedArray, StackedDistributedArray],
        x0: Union[DistributedArray, StackedDistributedArray],
        niter: int = 10,
        tol: float = 1e-4,
        show: bool = False,
        itershow: Tuple[int, int, int] = (10, 10, 10),
    ) -> Tuple[Union[DistributedArray, StackedDistributedArray], int, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        y : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Data of size (N,)
        x0 : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Initial guess of size (N,).
        niter : :obj:`int`, optional
            Number of iterations
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
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Estimated model of size (N,)
        iit : :obj:`int`
            Number of executed iterations
        cost : :obj:`numpy.ndarray`
            History of the L2 norm of the residual

        """

        x = self.setup(y=y, x0=x0, niter=niter, tol=tol, show=show)
        x = self.run(x, niter, show=show, itershow=itershow)
        self.finalize(show)
        return x, self.iiter, self.cost


class CGLS(Solver):
    r"""Conjugate gradient least squares

    Solve an overdetermined system of equations given either an MPILinearOperator or an MPIStackedLinearOperator ``Op``
    and distributed data ``y`` using conjugate gradient iterations.

    Parameters
    ----------
    Op : :obj:`pylops_mpi.MPILinearOperator` or :obj:`pylops_mpi.MPIStackedLinearOperator`
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
        sys.stdout.flush()

    def _print_step(self, x: Union[DistributedArray, StackedDistributedArray]) -> None:
        if isinstance(x, StackedDistributedArray):
            x = x.distarrays[0]
        strx = (
            f"{x[0]:1.2e}   "
            if np.iscomplexobj(x.local_array)
            else f"{x[0]:11.4e}        "
        )
        msg = (
            f"{self.iiter:6g}       "
            + strx
            + f"{self.cost[self.iiter]:11.4e}    {self.cost1[self.iiter]:11.4e}"
        )
        print(msg)
        sys.stdout.flush()

    def memory_usage(self) -> None:
        pass

    def setup(
        self,
        y: Union[DistributedArray, StackedDistributedArray],
        x0: Union[DistributedArray, StackedDistributedArray],
        niter: Optional[int] = None,
        damp: float = 0.0,
        tol: float = 1e-4,
        show: bool = False,
    ) -> Union[DistributedArray, StackedDistributedArray]:
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Data of size :math:`[N \times 1]`
        x0 : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Initial guess  of size (M,).
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
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Initial guess of size (N,).

        """
        self.y = y
        self.damp = damp**2
        self.tol = tol
        self.niter = niter

        x = x0.copy()
        # residual computed in-place into the output of the matvec
        self.s = self.Op.matvec(x)
        self.y.subtract(self.s, out=self.s)
        r = self.Op.rmatvec(self.s)
        if damp != 0.0:
            r -= x * damp
        self.rank = x.rank
        self.c = r.copy()
        self.q = self.Op.matvec(self.c)
        # r is kept as scratch space for the steps
        self.r = r
        self.kold = float(np.abs(r.dot(r, vdot=True)))

        # create variables to track the residual norm and iterations
        self.cost = []
        self.cost1 = []
        self.cost.append(float(self.s.norm()))
        if damp == 0.0:
            self.cost1.append(self.cost[0])
        else:
            self.cost1.append(
                np.sqrt(float(self.cost[0] ** 2 + damp * np.abs(x.dot(x, vdot=True))))
            )
        self.iiter = 0

        # print setup
        if show and self.rank == 0:
            if isinstance(x, StackedDistributedArray):
                is_complex = any(np.iscomplexobj(x1.local_array) for x1 in x.distarrays)
                self._print_setup(is_complex)
            else:
                self._print_setup(np.iscomplexobj(x.local_array))
        return x

    def step(
        self, x: Union[DistributedArray, StackedDistributedArray], show: bool = False
    ) -> Union[DistributedArray, StackedDistributedArray]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Current model vector to be updated by a step of CG
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Updated model vector

        """

        qq = self.q.dot(self.q, vdot=True)
        if self.damp != 0.0:
            qq += self.damp * self.c.dot(self.c, vdot=True)
        a = float(np.abs(self.kold / qq))
        # all updates are done in-place: the previous r is dead after the last
        # update of c, so it is used as scratch space; q is replaced by the
        # next matvec
        self.c.multiply(a, out=self.r)
        x += self.r
        self.q *= a
        self.s -= self.q
        r = self.Op.rmatvec(self.s)
        if self.damp != 0.0:
            x.multiply(self.damp, out=self.r)
            r -= self.r
        k = float(np.abs(r.dot(r, vdot=True)))
        b = float(k / self.kold)
        self.c *= b
        self.c += r
        self.r = r
        self.q = self.Op.matvec(self.c)
        self.kold = k
        self.iiter += 1
        self.cost.append(float(self.s.norm()))
        if self.damp == 0.0:
            self.cost1.append(self.cost[self.iiter])
        else:
            self.cost1.append(
                np.sqrt(
                    float(
                        self.cost[self.iiter] ** 2
                        + self.damp * np.abs(x.dot(x, vdot=True))
                    )
                )
            )
        if show and self.rank == 0:
            self._print_step(x)
        return x

    def run(
        self,
        x: Union[DistributedArray, StackedDistributedArray],
        niter: Optional[int] = None,
        show: bool = False,
        itershow: Tuple[int, int, int] = (10, 10, 10),
    ) -> Union[DistributedArray, StackedDistributedArray]:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
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
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray
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
        if show and self.rank == 0:
            self._print_finalize(nbar=65)
        self.cost = np.array(self.cost)

    def solve(
        self,
        y: Union[DistributedArray, StackedDistributedArray],
        x0: Union[DistributedArray, StackedDistributedArray],
        niter: int = 10,
        damp: float = 0.0,
        tol: float = 1e-4,
        show: bool = False,
        itershow: Tuple[int, int, int] = (10, 10, 10),
    ) -> Tuple[DistributedArray, int, int, float, float, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        y : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Data of size (N, )
        x0 : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Initial guess  of size (M, ).
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
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
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
