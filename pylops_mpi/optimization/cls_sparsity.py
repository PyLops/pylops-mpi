import time
import logging
from math import sqrt
from typing import Optional, Dict, Any, Callable, Union

import numpy as np

from pylops.optimization.basesolver import Solver
from pylops.optimization.cls_sparsity import _halfthreshold, _hardthreshold, _softthreshold
from pylops.utils import get_array_module, get_module_name, get_real_dtype
from pylops.utils.typing import NDArray, Tuple

from pylops_mpi.DistributedArray import DistributedArray, StackedDistributedArray
from pylops_mpi.LinearOperator import MPILinearOperator
from pylops_mpi.StackedLinearOperator import MPIStackedLinearOperator
from pylops_mpi.optimization.eigs import power_iteration

logger = logging.getLogger(__name__)


def _apply_thresh(x: Union[DistributedArray, StackedDistributedArray], threshf: Callable, thresh: float):
    """Apply thresholding

    Apply a thresholding function to a distributed array or stacked distributed array.

    Parameters
    ----------
    x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
        Input distributed array on which the thresholding operation is applied.
    threshf : :obj:`callable`
        Thresholding function with signature ``threshf(array, thresh)`` that
        operates on a local NumPy/CuPy array and returns a thresholded array
    thresh : :obj:`float`
        Threshold value passed to ``threshf``

    Returns
    -------
    x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
        The input array after applying the thresholding operation.
    """
    if isinstance(x, DistributedArray):
        x[:] = threshf(x.local_array, thresh)
    else:
        for iarr in range(x.narrays):
            x[iarr][:] = threshf(x[iarr].local_array, thresh)
    return x


class ISTA(Solver):
    r"""Iterative Shrinkage-Thresholding Algorithm (ISTA).

    Solve an optimization problem with :math:`L_p, \; p=0, 0.5, 1`
    regularization, given the operator ``Op`` and data ``y``. The operator
    can be real or complex, and should ideally be either square :math:`N=M`
    or underdetermined :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops_mpi.MPILinearOperator` or :obj:`pylops_mpi.StackedMPILinearOperator`
        Operator to invert

    Attributes
    ----------
    ncp : :obj:`module`
        Array module used by the solver (obtained via
        :func:`pylops.utils.backend.get_array_module`)
        ). Available only after ``setup`` is called.
    Opmatvec : :obj:`callable`
        Function handle to ``Op.matvec`` or ``Op.matmat``
        depending on the number of dimensions of ``y``.
    Oprmatvec : :obj:`callable`
        Function handle to ``Op.rmatvec`` or ``Op.rmatmat``
        depending on the number of dimensions of ``y``.
    SOpmatvec : :obj:`callable`
        Function handle to ``SOp.matvec`` or ``SOp.matmat``
        depending on the number of dimensions of ``y``.
    SOprmatvec : :obj:`callable`
        Function handle to ``SOp.rmatvec`` or ``SOp.rmatmat``
        depending on the number of dimensions of ``y``.
    threshf : :obj:`callable`
        Function handle to the chosen thresholding method.
    thresh : :obj:`float`
        Threshold.
    normresold : :obj:`float`
        Old norm of the residual.
    t : :obj:`float`
        FISTA auxiliary coefficient (not used in ISTA).
    cost : :obj:`list`
        History of the L2 norm of the total objectiv function. Available
        only after ``setup`` is called and updated at each call to ``step``.
    iiter : :obj:`int`
        Current iteration number. Available only after
        ``setup`` is called and updated at each call to ``step``.

    Raises
    ------
    NotImplementedError
        If ``threshkind`` is different from hard, soft, half
    """

    def _print_setup(self) -> None:
        self._print_solver(f" ({self.threshkind} thresholding)")
        if self.niter is not None:
            strpar = f"eps = {self.eps:10e}\ttol = {self.tol:10e}\tniter = {self.niter}"
        else:
            strpar = f"eps = {self.eps:10e}\ttol = {self.tol:10e}"
        strpar1 = f"alpha = {self.alpha:10e}\tthresh = {self.thresh:10e}"
        head1 = "   Itn          x[0]              r2norm     r12norm     xupdate"
        print(strpar)
        print(strpar1)
        print("-" * 80)
        print(head1)

    def _print_step(
            self,
            x: Union[DistributedArray, StackedDistributedArray],
            costdata: float,
            costreg: float,
            xupdate: float,
    ) -> None:
        if isinstance(x, StackedDistributedArray):
            x = x.distarrays[0]
        strx = (
            f"  {x[0]:1.2e}   " if np.iscomplexobj(x.local_array) else f"     {x[0]:11.4e}        "
        )
        msg = (
            f"{self.iiter:6g} "
            + strx
            + f"{costdata:10.3e}   {costdata + costreg:9.3e}  {xupdate:10.3e}"
        )
        print(msg)

    def memory_usage(
            self,
            show: bool = False,
            unit: str = "B",
    ) -> float:
        pass

    def setup(
            self,
            y: Union[DistributedArray, StackedDistributedArray],
            x0: Union[DistributedArray, StackedDistributedArray],
            niter: Optional[int] = None,
            SOp: Optional[Union[MPILinearOperator, MPIStackedLinearOperator]] = None,
            eps: float = 0.1,
            alpha: Optional[float] = None,
            eigsdict: Optional[Dict[str, Any]] = None,
            tol: float = 1e-10,
            threshkind: str = "soft",
            decay: Optional[NDArray] = None,
            monitorres: bool = False,
            show: bool = False,
    ) -> DistributedArray:
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Data of size (N, )
        x0 : :obj:`pylops_mpi.DistributedArray`` or :obj:`pylops_mpi.StackedDistributedArray`
            Initial guess  of size (M, ).
        niter : :obj:`int`
            Number of iterations
        SOp : :obj:`pylops_mpi.MPILinearOperator` or :obj:`pylops_mpi.MPIStackedLinearOperator`, optional
            Regularization operator (use when solving the analysis problem)
        eps : :obj:`float`, optional
            Sparsity damping
        alpha : :obj:`float`, optional
            Step size. To guarantee convergence, ensure
            :math:`\alpha \le 1/\lambda_\text{max}`, where :math:`\lambda_\text{max}`
            is the largest eigenvalue of :math:`\mathbf{Op}^H\mathbf{Op}`.
            If ``None``, the maximum eigenvalue is estimated and the optimal step size
            is chosen as :math:`1/\lambda_\text{max}`. If provided, the
            convergence criterion will not be checked internally.
        eigsdict : :obj:`dict`, optional
            Dictionary of parameters to be passed to power_iteration` method
            when computing the maximum eigenvalue
        tol : :obj:`float`, optional
            Tolerance. Stop iterations if difference between inverted model
            at subsequent iterations is smaller than ``tol``
        threshkind : :obj:`str`, optional
            Kind of thresholding ('hard', 'soft', 'half' - 'soft' used as default)
        decay : :obj:`numpy.ndarray`, optional
            Decay factor to be applied to thresholding during iterations
        monitorres : :obj:`bool`, optional
            Monitor that residual is decreasing
        show : :obj:`bool`, optional
            Display setup log

        Returns
        -------
        x : :obj:`pylops_mpi.DistributedArray`
            Initial guess of size (N,).

        """

        self.y = y
        self.SOp = SOp
        self.niter = niter
        self.eps = eps
        self.eigsdict = {} if eigsdict is None else eigsdict
        self.tol = tol
        self.threshkind = threshkind
        self.decay = decay
        self.monitorres = monitorres
        if isinstance(y, StackedDistributedArray):
            self.ncp = get_array_module(y[0].local_array)
        else:
            self.ncp = get_array_module(y.local_array)

        self.Opmatvec = self.Op.matvec
        self.Oprmatvec = self.Op.rmatvec
        if self.SOp is not None:
            self.SOpmatvec = self.SOp.matvec
            self.SOprmatvec = self.SOp.rmatvec

        if threshkind not in [
            "hard",
            "soft",
            "half",
        ]:
            raise ValueError(
                f"threshkind must be hard, soft, half, got {threshkind}"
            )

        self.threshf: Callable[[DistributedArray, float], DistributedArray]
        if threshkind == "soft":
            self.threshf = _softthreshold
        elif threshkind == "hard":
            self.threshf = _hardthreshold
        elif threshkind == "half":
            self.threshf = _halfthreshold

        if decay is None:
            self.decay = self.ncp.ones(niter, dtype=get_real_dtype(self.Op.dtype))

        # step size
        if alpha is not None:
            self.alpha = alpha
        elif not hasattr(self, "alpha"):
            # compute largest eigenvalues of Op^H * Op
            Op1 = self.Op.H @ self.Op
            maxeig = np.abs(
                power_iteration(
                    Op1,
                    b_k=x0.empty_like(),
                    dtype=Op1.dtype,
                    backend=get_module_name(self.ncp),
                    **self.eigsdict
                )[0]
            )
            print("maxeieur", maxeig)
            self.alpha = float(1.0 / maxeig)
        self.thresh = eps * self.alpha * 0.5
        x = x0.copy()

        # create variable to track residual
        if monitorres:
            self.normresold = np.inf
        self.t = 1.0

        # create variables to track the residual norm and iterations
        self.cost = []
        self.iiter = 0
        if show:
            self._print_setup()
        return x

    def step(
            self,
            x: Union[DistributedArray, StackedDistributedArray],
            show: bool = False
    ) -> (Tuple)[Union[DistributedArray, StackedDistributedArray], float]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Current model vector to be updated by a step of ISTA
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Updated model vector
        xupdate : :obj:`float`
            Norm of the update

        """
        # store old vector
        xold = x.copy()

        # compute residual
        res = self.y - self.Opmatvec(x)

        if self.monitorres:
            self.normres = res.norm()
            if self.normres > self.normresold:
                raise ValueError(
                    f"ISTA stopped at iteration {self.iiter} due to "
                    "residual increasing, consider modifying "
                    "eps and/or alpha..."
                )
            else:
                self.normresold = self.normres

        # compute gradient
        grad = self.alpha * (self.Oprmatvec(res))

        # update inverted model
        x_unthesh = x + grad

        # apply SOp.H to current x
        if self.SOp is not None:
            SOpx_unthesh = self.SOprmatvec(x_unthesh)

        # threshold current solution or current solution projected onto SOp.H space
        if self.SOp is None:
            x_unthesh_or_SOpx_unthesh = x_unthesh
        else:
            x_unthesh_or_SOpx_unthesh = SOpx_unthesh

        x = _apply_thresh(
            x_unthesh_or_SOpx_unthesh, self.threshf,
            self.decay[self.iiter] * self.thresh,
        )

        # apply SOp to thresholded x
        if self.SOp is not None:
            x = self.SOpmatvec(x)

        # compute model update norm
        xupdate = (x - xold).norm().item()

        costdata = 0.5 * res.norm().item() ** 2
        costreg = self.eps * x.norm(ord=1).item()
        self.cost.append(float(costdata + costreg))
        self.iiter += 1
        if show:
            self._print_step(x, costdata, costreg, xupdate)
        return x, xupdate

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
            Current model vector to be updated by multiple steps of ISTA
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
            Estimated model of size (M, ).

        """

        xupdate = np.inf
        niter = self.niter if niter is None else niter
        if niter is None:
            raise ValueError("niter must not be None")
        while self.iiter < niter and xupdate > self.tol:
            showstep = (
                True
                if show and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x, xupdate = self.step(x, showstep)
            self.callback(x)
        if xupdate <= self.tol:
            logger.info("Update smaller that tolerance for iteration %d", self.iiter)
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
        if show:
            self._print_finalize()

    def solve(
            self,
            y: Union[DistributedArray, StackedDistributedArray],
            x0: Union[DistributedArray, StackedDistributedArray],
            niter: Optional[int] = None,
            SOp: Optional[MPILinearOperator] = None,
            eps: float = 0.1,
            alpha: Optional[float] = None,
            eigsdict: Optional[Dict[str, Any]] = None,
            tol: float = 1e-10,
            threshkind: str = "soft",
            decay: Optional[DistributedArray] = None,
            monitorres: bool = False,
            show: bool = False,
            itershow: Tuple[int, int, int] = (10, 10, 10),
    ) -> Tuple[Union[DistributedArray, StackedDistributedArray], int, NDArray]:
        r"""
        Parameters
        ----------
        y : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Data of size (N, )
        x0 : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Initial guess  of size (M, ).
        niter : :obj:`int`
            Number of iterations
        SOp : :obj:`pylops_mpi.MPILinearOperator` or :obj:`pylops_mpi.MPIStackedLinearOperator`, optional
            Regularization operator (use when solving the analysis problem)
        eps : :obj:`float`, optional
            Sparsity damping
        alpha : :obj:`float`, optional
            Step size. To guarantee convergence, ensure
            :math:`\alpha \le 1/\lambda_\text{max}`, where :math:`\lambda_\text{max}`
            is the largest eigenvalue of :math:`\mathbf{Op}^H\mathbf{Op}`.
            If ``None``, the maximum eigenvalue is estimated and the optimal step size
            is chosen as :math:`1/\lambda_\text{max}`. If provided, the
            convergence criterion will not be checked internally.
        eigsdict : :obj:`dict`, optional
            Dictionary of parameters to be passed to power_iteration method
            when computing the maximum eigenvalue
        tol : :obj:`float`, optional
            Tolerance. Stop iterations if difference between inverted model
            at subsequent iterations is smaller than ``tol``
        threshkind : :obj:`str`, optional
            Kind of thresholding ('hard', 'soft', 'half' - 'soft' used as default)
        decay : :obj:`numpy.ndarray`, optional
            Decay factor to be applied to thresholding during iterations
        monitorres : :obj:`bool`, optional
            Monitor that residual is decreasing
        show : :obj:`bool`, optional
            Display setup log

        Returns
        -------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Initial guess of size (N,).

        """
        x = self.setup(
            y=y,
            x0=x0,
            niter=niter,
            SOp=SOp,
            eps=eps,
            alpha=alpha,
            eigsdict=eigsdict,
            tol=tol,
            threshkind=threshkind,
            decay=decay,
            monitorres=monitorres,
            show=show,
        )
        x = self.run(x, niter, show=show, itershow=itershow)
        self.finalize(show)
        return x, self.iiter, self.cost


class FISTA(ISTA):
    r"""Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).

    Solve an optimization problem with :math:`L_p, \; p=0, 0.5, 1`
    regularization, given the operator ``Op`` and data ``y``.
    The operator can be real or complex, and should ideally be either square
    :math:`N=M` or underdetermined :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops_mpi.MPILinearOperator` or :obj:`pylops_mpi.MPIStackedLinearOperator`
        Operator to invert


    Attributes
    ----------
    ncp : :obj:`module`
        Array module used by the solver (obtained via
        :func:`pylops.utils.backend.get_array_module`)
        ). Available only after ``setup`` is called.
    Opmatvec : :obj:`callable`
        Function handle to ``Op.matvec`` or ``Op.matmat``
        depending on the number of dimensions of ``y``.
    Oprmatvec : :obj:`callable`
        Function handle to ``Op.rmatvec`` or ``Op.rmatmat``
        depending on the number of dimensions of ``y``.
    SOpmatvec : :obj:`callable`
        Function handle to ``SOp.matvec`` or ``SOp.matmat``
        depending on the number of dimensions of ``y``.
    SOprmatvec : :obj:`callable`
        Function handle to ``SOp.rmatvec`` or ``SOp.rmatmat``
        depending on the number of dimensions of ``y``.
    threshf : :obj:`callable`
        Function handle to the chosen thresholding method.
    thresh : :obj:`float`
        Threshold.
    normresold : :obj:`float`
        Old norm of the residual.
    t : :obj:`float`
        FISTA auxiliary coefficient (not used in ISTA).
    cost : :obj:`list`
        History of the L2 norm of the total objectiv function. Available
        only after ``setup`` is called and updated at each call to ``step``.
    iiter : :obj:`int`
        Current iteration number. Available only after
        ``setup`` is called and updated at each call to ``step``.

    Raises
    ------
    NotImplementedError
        If ``threshkind`` is different from hard, soft, half.

    See Also
    --------
    ISTA: Iterative Shrinkage-Thresholding Algorithm (ISTA).

    Notes
    -----
    Solves the following synthesis problem for the operator
    :math:`\mathbf{Op}` and the data :math:`\mathbf{y}`:

    .. math::
        J = \|\mathbf{y} - \mathbf{Op}\,\mathbf{x}\|_2^2 +
            \epsilon \|\mathbf{x}\|_p

    or the analysis problem:

    .. math::
        J = \|\mathbf{y} - \mathbf{Op}\,\mathbf{x}\|_2^2 +
            \epsilon \|\mathbf{SOp}^H\,\mathbf{x}\|_p

    if ``SOp`` is provided.

    The Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) [1]_ is used,
    where :math:`p=0, 0.5, 1`. This is a modified version of ISTA solver with
    improved convergence properties and limited additional computational cost.
    Similarly to the ISTA solver, the choice of the thresholding algorithm to
    apply at every iteration is based on the choice of :math:`p`.

    .. [1] Beck, A., and Teboulle, M., “A Fast Iterative Shrinkage-Thresholding
       Algorithm for Linear Inverse Problems”, SIAM Journal on
       Imaging Sciences, vol. 2, pp. 183-202. 2009.

    """

    def memory_usage(
        self,
        show: bool = False,
        unit: str = "B",
    ) -> float:
        pass

    def step(
        self,
        x: Union[DistributedArray, StackedDistributedArray],
        z: Union[DistributedArray, StackedDistributedArray],
        show: bool = False
    ) -> Tuple[Union[DistributedArray, StackedDistributedArray], Union[DistributedArray, StackedDistributedArray], float]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Current model vector to be updated by a step of FISTA
        z : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Current auxiliary model vector to be updated by a step of FISTA
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Updated model vector
        z : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Updated auxiliary model vector
        xupdate : :obj:`float`
            Norm of the update

        """
        # store old vector
        xold = x.copy()

        # compute residual
        res = self.y - self.Opmatvec(z)

        if self.monitorres:
            self.normres = res.norm()
            if self.normres > self.normresold:
                raise ValueError(
                    f"FISTA stopped at iteration {self.iiter} due to "
                    "residual increasing, consider modifying "
                    "eps and/or alpha..."
                )
            else:
                self.normresold = self.normres

        # compute gradient and update inverted model
        grad = self.alpha * (self.Oprmatvec(res))
        x_unthesh = z + grad

        # apply SOp.H to current x
        if self.SOp is not None:
            SOpx_unthesh = self.SOprmatvec(x_unthesh)

        # threshold current solution or current solution projected onto SOp.H space
        if self.SOp is None:
            x_unthesh_or_SOpx_unthesh = x_unthesh
        else:
            x_unthesh_or_SOpx_unthesh = SOpx_unthesh
        x = _apply_thresh(
            x_unthesh_or_SOpx_unthesh, self.threshf,
            self.decay[self.iiter] * self.thresh,
        )

        # apply SOp to thresholded x
        if self.SOp is not None:
            x = self.SOpmatvec(x)

        # update auxiliary coefficients
        told = self.t
        self.t = (1.0 + sqrt(1.0 + 4.0 * self.t**2)) / 2.0

        # model update
        z = x + ((told - 1.0) / self.t) * (x - xold)

        # check model update
        xupdate = (x - xold).norm().item()

        # cost functions
        costdata = 0.5 * (self.y - self.Op @ x).norm().item() ** 2
        costreg = self.eps * x.norm(ord=1).item()
        self.cost.append(float(costdata + costreg))

        self.iiter += 1
        if show:
            self._print_step(x, costdata, costreg, xupdate)
        return x, z, xupdate

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
            Current model vector to be updated by multiple steps of FISTA
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
            Estimated model of size (M,).

        """
        z = x.copy()
        xupdate = np.inf
        niter = self.niter if niter is None else niter
        if niter is None:
            raise ValueError("niter must not be None")
        while self.iiter < niter and xupdate > self.tol:
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
            x, z, xupdate = self.step(x, z, showstep)
            self.callback(x)
        if xupdate <= self.tol:
            logger.warning(
                "Update smaller that tolerance for " "iteration %d", self.iiter
            )
        return x
