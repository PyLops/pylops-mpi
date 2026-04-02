import time
import logging
import numpy as np
from typing import Optional, Dict, Any, Callable

from pylops.optimization.basesolver import Solver
from pylops.optimization.callback import _callback_stop
from pylops.utils import get_array_module, get_module_name, get_real_dtype
from pylops.utils.typing import NDArray, Union, Tuple

from pylops_mpi.DistributedArray import DistributedArray, StackedDistributedArray
from pylops_mpi.LinearOperator import MPILinearOperator
from pylops_mpi.optimization.eigs import power_iteration

logger = logging.getLogger(__name__)


def _hardthreshold(x: DistributedArray, thresh: float) -> DistributedArray:
    r"""Hard thresholding.

    Applies hard thresholding to vector ``x`` (equal to the proximity
    operator for :math:`\|\mathbf{x}\|_0`) as shown in [1]_.

    .. [1] Chen, F., Shen, L., Suter, B.W., “Computing the proximity
        operator of the ℓp norm with 0 < p < 1”,
        IET Signal Processing, vol. 10. 2016.

    Parameters
    ----------
    x : :obj:`pylops_mpi.DistributedArray`
        Vector
    thresh : :obj:`float`
        Threshold

    Returns
    -------
    x1 : :obj:`pylops_mpi.DistributedArray`
        Threshold vector
    """
    local = x.local_array
    local_out = local.copy()
    # Set elements to 0 where their magnitude is less than or equal to sqrt(2 * thresh) at each rank
    local_out[np.abs(local) <= np.sqrt(2 * thresh)] = 0.0

    dist = DistributedArray(global_shape=x.global_shape, partition=x.partition, base_comm=x.base_comm,
                            base_comm_nccl=x.base_comm_nccl, dtype=x.dtype, engine=x.engine)
    dist[:] = local_out
    return dist


def _softthreshold(x: DistributedArray, thresh: float) -> DistributedArray:
    r"""Soft thresholding.

    Applies soft thresholding to vector ``x`` (equal to the proximity
    operator for :math:`\|\mathbf{x}\|_1`) as shown in [1]_.

    .. [1] Chen, F., Shen, L., Suter, B.W., “Computing the proximity
       operator of the ℓp norm with 0 < p < 1”,
       IET Signal Processing, vol. 10. 2016.

    Parameters
    ----------
    x : :obj:`pylops_mpi.DistributedArray`
        Vector
    thresh : :obj:`float`
        Threshold

    Returns
    -------
    x1 : :obj:`pylops_mpi.DistributedArray`
        Threshold vector

    """
    x_local = x.local_array
    if np.iscomplexobj(x_local):
        x1 = np.maximum(np.abs(x_local) - thresh, 0.0) * np.exp(1j * np.angle(x_local))
    else:
        x1 = np.maximum(np.abs(x_local) - thresh, 0.0) * np.sign(x_local)
    dist = DistributedArray(global_shape=x.global_shape, partition=x.partition, base_comm=x.base_comm,
                            base_comm_nccl=x.base_comm_nccl, dtype=x.dtype, engine=x.engine)
    dist[:] = x1
    return dist


def _halfthreshold(x: DistributedArray, thresh: float) -> DistributedArray:
    r"""Half thresholding.

    Applies half thresholding to vector ``x`` (equal to the proximity
    operator for :math:`\|\mathbf{x}\|_{1/2}^{1/2}`) as shown in [1]_.

    .. [1] Chen, F., Shen, L., Suter, B.W., “Computing the proximity
       operator of the ℓp norm with 0 < p < 1”,
       IET Signal Processing, vol. 10. 2016.

    Parameters
    ----------
    x : :obj:`pylops_mpi.DistributedArray`
        Vector
    thresh : :obj:`float`
        Threshold

    Returns
    -------
    x1 : :obj:`pylops_mpi.DistributedArray`
        Threshold vector
    """
    # Get the local array
    local = x.local_array
    ncp = get_array_module(local)
    arg = ncp.ones_like(local)
    # Create a mask to avoid division by zero
    mask = local != 0
    arg[mask] = (thresh / 8.0) * (ncp.abs(local[mask]) / 3.0) ** (-1.5)

    if ncp.iscomplexobj(arg):
        arg.real = ncp.clip(arg.real, -1.0, 1.0)
        arg.imag = ncp.clip(arg.imag, -1.0, 1.0)
    else:
        arg = ncp.clip(arg, -1.0, 1.0)

    phi = 2.0 / 3.0 * ncp.arccos(arg)

    # Apply the non-linear shrinkage transformation:
    local_out = 2.0 / 3.0 * local * (1.0 + ncp.cos(2.0 * np.pi / 3.0 - phi))

    # Threshold = (54^(1/3) / 4) * thresh^(2/3)
    local_out[ncp.abs(local) <= (54.0 ** (1.0 / 3.0) / 4.0) * thresh ** (2.0 / 3.0)] = 0
    dist = DistributedArray(global_shape=x.global_shape, partition=x.partition, base_comm=x.base_comm,
                            base_comm_nccl=x.base_comm_nccl, dtype=x.dtype, engine=x.engine)
    dist[:] = local_out
    return dist


class ISTA(Solver):
    r"""Iterative Shrinkage-Thresholding Algorithm (ISTA).

    Solve an optimization problem with :math:`L_p, \; p=0, 0.5, 1`
    regularization, given the operator ``Op`` and data ``y``. The operator
    can be real or complex, and should ideally be either square :math:`N=M`
    or underdetermined :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops_mpi.MPILinearOperator`
        Operator to invert

    Attributes
    ----------
    ncp : :obj:`module`
        Array module used by the solver (obtained via
        :func:`pylops.utils.backend.get_array_module`)
        ). Available only after ``setup`` is called.
    isjax : :obj:`bool`
        Whether the input data is a JAX array or not.
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
    res : :obj:`numpy.ndarray`
        Residual vector of size :math:`[N \times 1]` used in the
        solver when ``preallocate=True``. Available only after ``setup``
        is called and updated at each call to ``step``.
    grad : :obj:`numpy.ndarray`
        Gradient vector of size :math:`[M \times 1]` used in the
        solver when ``preallocate=True``. Available only after ``setup``
        is called and updated at each call to ``step``.
    x_unthesh : :obj:`numpy.ndarray`
        Unthresholded model vector of size :math:`[M \times 1]` used in the
        solver when ``preallocate=True``. Available only after ``setup``
        is called and updated at each call to ``step``.
    xold : :obj:`numpy.ndarray`
        Old model vector of size :math:`[M \times 1]` used in the
        solver when ``preallocate=True``. Available only after ``setup``
        is called and updated at each call to ``step``.
    SOpx_unthesh : :obj:`numpy.ndarray`
        Old model vector pre-multiplied by the regularization operator
        of size :math:`[M_S \times 1]` used in the solver when ``preallocate=True``.
        Available only after ``setup`` is called and updated at each call to ``step``.
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
        if self.perc is None:
            strpar1 = f"alpha = {self.alpha:10e}\tthresh = {self.thresh:10e}"
        else:
            strpar1 = f"alpha = {self.alpha:10e}\tperc = {self.perc:.1f}"
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
        if isinstance(x, DistributedArray):
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
            SOp: Optional[MPILinearOperator] = None,
            eps: float = 0.1,
            alpha: Optional[float] = None,
            eigsdict: Optional[Dict[str, Any]] = None,
            tol: float = 1e-10,
            threshkind: str = "soft",
            perc: Optional[float] = None,
            decay: Optional[NDArray] = None,
            monitorres: bool = False,
            preallocate: bool = False,
            show: bool = False,
    ) -> Union[DistributedArray, StackedDistributedArray]:
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Data of size (N, )
        x0 : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Initial guess  of size (M, ).
        niter : :obj:`int`
            Number of iterations
        SOp : :obj:`pylops_mpi.MPILinearOperator`, optional
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
        perc : :obj:`float`, optional
            Percentile, as percentage of values to be kept by thresholding (to be
            provided when thresholding is soft-percentile or half-percentile)
        decay : :obj:`numpy.ndarray`, optional
            Decay factor to be applied to thresholding during iterations
        monitorres : :obj:`bool`, optional
            Monitor that residual is decreasing
        preallocate : :obj:`bool`, optional
            Pre-allocate all variables used by the solver.
        show : :obj:`bool`, optional
            Display setup log

        Returns
        -------
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Initial guess of size (N,).

        """

        self.y = y
        self.SOp = SOp
        self.niter = niter
        self.eps = eps
        self.eigsdict = {} if eigsdict is None else eigsdict
        self.tol = tol
        self.threshkind = threshkind
        self.perc = perc
        self.decay = decay
        self.monitorres = monitorres
        self.preallocate = preallocate
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
                "threshkind must be hard, soft, half,"
            )

        self.threshf: Callable[[DistributedArray, float], DistributedArray]
        if threshkind == "soft":
            self.threshf = _softthreshold
        elif threshkind == "hard":
            self.threshf = _hardthreshold
        elif threshkind == "half":
            self.threshf = _halfthreshold

        if perc is None and decay is None:
            self.decay = self.ncp.ones(niter, dtype=get_real_dtype(self.Op.dtype))
        if alpha is not None:
            self.alpha = alpha
        elif not hasattr(self, "alpha"):
            Op1 = self.Op.H @ self.Op
            maxeig = np.abs(
                power_iteration(
                    Op1,
                    dtype=Op1.dtype,
                    backend=get_module_name(self.ncp),
                    **self.eigsdict
                )[0]
            )
            self.alpha = float(1.0 / maxeig)
        self.thresh = eps * self.alpha * 0.5
        if x0 is None:
            x = DistributedArray(global_shape=self.Op.shape[1], dtype=self.Op.dtype, engine=y.engine)
            x[:] = 0
        else:
            x = x0.copy()
        if self.preallocate:
            self.res = DistributedArray(global_shape=y.global_shape, dtype=y.dtype, engine=y.engine,
                                        base_comm=y.base_comm, base_comm_nccl=y.base_comm_nccl)
            self.res = DistributedArray(global_shape=x.global_shape, dtype=x.dtype, engine=x.engine,
                                        base_comm=x.base_comm, base_comm_nccl=x.base_comm_nccl)
            self.x_unthesh = DistributedArray(global_shape=x.global_shape, dtype=x.dtype, engine=x.engine,
                                              base_comm=x.base_comm, base_comm_nccl=x.base_comm_nccl)
            self.xold = DistributedArray(global_shape=x.global_shape, dtype=x.dtype, engine=x.engine,
                                         base_comm=x.base_comm, base_comm_nccl=x.base_comm_nccl)
            if self.SOp is not None:
                self.SOpx_unthesh = DistributedArray(global_shape=self.SOp.shape[1], dtype=self.SOp.dtype,
                                                     engine=x.engine, base_comm=x.base_comm,
                                                     base_comm_nccl=x.base_comm_nccl)
                self.SOpx_unthesh[:] = 0
        if monitorres:
            self.normresold = np.inf
        self.t = 1.0
        self.cost = []
        self.iiter = 0
        if show:
            self._print_setup()
        return x

    def step(self, x: Union[DistributedArray, StackedDistributedArray], show: bool = False) -> (
            Tuple)[Union[DistributedArray, StackedDistributedArray], float]:
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
        if self.preallocate:
            self.xold[:] = x[:]
        else:
            xold = x.copy()
        if not self.preallocate:
            res = self.y - self.Opmatvec(x)
        else:
            self.res = self.y - self.Opmatvec(x)

        if self.monitorres:
            self.normres = self.res.norm() if self.preallocate else res.norm()
            if self.normres > self.normresold:
                raise ValueError(
                    f"ISTA stopped at iteration {self.iiter} due to "
                    "residual increasing, consider modifying "
                    "eps and/or alpha..."
                )
            else:
                self.normresold = self.normres
        if not self.preallocate:
            grad = self.alpha * (self.Oprmatvec(res))
        else:
            self.grad = self.alpha * (self.Oprmatvec(self.res))

        if not self.preallocate:
            x_unthesh = x + grad
        else:
            self.x_unthesh = x + self.grad

        if self.SOp is not None:
            if self.preallocate:
                self.SOpx_unthesh[:] = self.SOprmatvec(self.x_unthesh)
            else:
                SOpx_unthesh = self.SOprmatvec(x_unthesh)

        if self.SOp is None:
            x_unthesh_or_SOpx_unthesh = (
                self.x_unthesh if self.preallocate else x_unthesh
            )
        else:
            self.x_unthesh_or_SOpx_unthesh = (
                self.SOpx_unthesh if self.preallocate else SOpx_unthesh
            )

        if self.perc is None:
            x = self.threshf(
                x_unthesh_or_SOpx_unthesh,
                self.decay[self.iiter] * self.thresh,
            )
        else:
            x = self.threshf(x_unthesh_or_SOpx_unthesh, 100 - self.perc)

        if self.SOp is not None:
            x = self.SOpmatvec(x)

        if not self.preallocate:
            xupdate = x - xold
            xupdate = xupdate.norm()
        else:
            self.xold = x - self.xold
            xupdate = self.xold.norm()

        costdata = (0.5 * self.res.norm() if self.preallocate else res.norm() ** 2).item()
        costreg = (self.eps * x.norm(ord=1)).item()
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
        x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray
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
            # check if any callback has raised a stop flag
            stop = _callback_stop(self.callbacks)
            if stop:
                break
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
            x0: Optional[Union[DistributedArray, StackedDistributedArray]] = None,
            niter: Optional[int] = None,
            SOp: Optional[MPILinearOperator] = None,
            eps: float = 0.1,
            alpha: Optional[float] = None,
            eigsdict: Optional[Dict[str, Any]] = None,
            tol: float = 1e-10,
            threshkind: str = "soft",
            perc: Optional[float] = None,
            decay: Optional[DistributedArray] = None,
            monitorres: bool = False,
            preallocate: bool = False,
            show: bool = False,
            itershow: Tuple[int, int, int] = (10, 10, 10),
    ) -> Tuple[DistributedArray, int, NDArray]:
        r"""
        Parameters
        ----------
        y : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Data of size (N, )
        x0 : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
            Initial guess  of size (M, ).
        niter : :obj:`int`
            Number of iterations
        SOp : :obj:`pylops_mpi.MPILinearOperator`, optional
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
        perc : :obj:`float`, optional
            Percentile, as percentage of values to be kept by thresholding (to be
            provided when thresholding is soft-percentile or half-percentile)
        decay : :obj:`numpy.ndarray`, optional
            Decay factor to be applied to thresholding during iterations
        monitorres : :obj:`bool`, optional
            Monitor that residual is decreasing
        preallocate : :obj:`bool`, optional
            Pre-allocate all variables used by the solver.
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
            perc=perc,
            decay=decay,
            monitorres=monitorres,
            preallocate=preallocate,
            show=show,
        )
        x = self.run(x, niter, show=show, itershow=itershow)
        self.finalize(show)
        return x, self.iiter, self.cost
