from typing import Optional, Dict, Any, Callable

from pylops.utils.typing import Tuple, NDArray
from pylops.optimization.callback import CostNanInfCallback, CostToInitialCallback, CostToDataCallback

from pylops_mpi.DistributedArray import DistributedArray
from pylops_mpi.LinearOperator import MPILinearOperator
from pylops_mpi.optimization.cls_sparsity import ISTA, FISTA


def ista(
        Op: MPILinearOperator,
        y: DistributedArray,
        x0: Optional[DistributedArray],
        niter: int = 10,
        SOp: Optional[MPILinearOperator] = None,
        eps: float = 0.1,
        alpha: Optional[float] = None,
        eigsdict: Optional[Dict[str, Any]] = None,
        tol: float = 1e-10,
        rtol: float = 0.0,
        rtol1: float = 0.0,
        threshkind: str = "soft",
        decay: Optional[NDArray] = None,
        monitorres: bool = False,
        show: bool = False,
        itershow: Tuple[int, int, int] = (10, 10, 10),
        callback: Optional[Callable] = None,
) -> Tuple[DistributedArray, int, NDArray]:
    r"""Iterative Shrinkage-Thresholding Algorithm (ISTA).

    Solve an optimization problem with :math:`L^p, \; p=0, 0.5, 1`
    regularization, given the operator ``Op`` and data ``y``. The operator
    can be real or complex, and should ideally be either square :math:`N=M`
    or underdetermined :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    y : :obj:`numpy.ndarray`
        Data of size :math:`[N \times 1]`
    x0: :obj:`numpy.ndarray`, optional
        Initial guess
    niter : :obj:`int`
        Number of iterations
    SOp : :obj:`pylops.LinearOperator`, optional
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
        Absolute tolerance on model update. Stop iterations if difference between inverted model
        at subsequent iterations is smaller than ``tol``
    rtol : :obj:`float`, optional
        Relative tolerance on total cost function wrt initial total cost
        function. Stops the solver when the ratio of the current total cost function
        to the initial total cost function is below this value.
    rtol1 : :obj:`float`, optional
        Relative tolerance on total cost function wrt to data. Stops the solver when
        the ratio of the current total cost function to the data norm is below this value.
    threshkind : :obj:`str`, optional
        Kind of thresholding ('hard', 'soft', 'half' - 'soft' used as default)
    decay : :obj:`numpy.ndarray`, optional
        Decay factor to be applied to thresholding during iterations
    monitorres : :obj:`bool`, optional
        Monitor that residual is decreasing
    show : :obj:`bool`, optional
        Display logs
    itershow : :obj:`tuple`, optional
        Display set log for the first N1 steps, last N2 steps,
        and every N3 steps in between where N1, N2, N3 are the
        three element of the list.
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    preallocate : :obj:`bool`, optional
        Pre-allocate all variables used by the solver.

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model
    niter : :obj:`int`
        Number of effective iterations
    cost : :obj:`numpy.ndarray`
        History of cost (including regularization term)

    Raises
    ------
    NotImplementedError
        If ``threshkind`` is different from hard, soft, half

    """
    callbacks = [
        CostNanInfCallback(),
    ]
    if rtol > 0.0:
        callbacks.append(CostToInitialCallback(rtol))
    if rtol1 > 0.0:
        callbacks.append(CostToDataCallback(rtol1))

    istasolve = ISTA(
        Op,
        callbacks=callbacks,
    )
    if callback is not None:
        istasolve.callback = callback
    x, iiter, cost = istasolve.solve(
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
        itershow=itershow,
    )
    return x, iiter, cost


def fista(
    Op: MPILinearOperator,
    y: DistributedArray,
    x0: Optional[DistributedArray],
    niter: int = 10,
    SOp: Optional[MPILinearOperator] = None,
    eps: float = 0.1,
    alpha: Optional[float] = None,
    eigsdict: Optional[Dict[str, Any]] = None,
    tol: float = 1e-10,
    rtol: float = 0.0,
    rtol1: float = 0.0,
    threshkind: str = "soft",
    decay: Optional[NDArray] = None,
    monitorres: bool = False,
    show: bool = False,
    itershow: Tuple[int, int, int] = (10, 10, 10),
    callback: Optional[Callable] = None,
) -> Tuple[DistributedArray, int, NDArray]:
    r"""Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).

    Solve an optimization problem with :math:`L^p, \; p=0, 0.5, 1`
    regularization, given the operator ``Op`` and data ``y``.
    The operator can be real or complex, and should ideally be either square
    :math:`N=M` or underdetermined :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops_mpi.MPILinearOperator`
        Operator to invert
    y : :obj:`pylops_mpi.DistributedArray`
        Data
    x0: :obj:`pylops_mpi.DistributedArray`, optional
        Initial guess
    niter : :obj:`int`, optional
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
        Dictionary of parameters to be passed to :func:`pylops.LinearOperator.eigs` method
        when computing the maximum eigenvalue
    tol : :obj:`float`, optional
        Absolute tolerance on model update. Stop iterations if difference between inverted model
        at subsequent iterations is smaller than ``tol``
    rtol : :obj:`float`, optional
        Relative tolerance on total cost function wrt initial total cost
        function. Stops the solver when the ratio of the current total cost function
        to the initial total cost function is below this value.
    rtol1 : :obj:`float`, optional
        Relative tolerance on total cost function wrt to data. Stops the solver when
        the ratio of the current total cost function to the data norm is below this value.
    threshkind : :obj:`str`, optional
        Kind of thresholding ('hard', 'soft', 'half' - 'soft' used as default)
    decay : :obj:`numpy.ndarray`, optional
        Decay factor to be applied to thresholding during iterations
    monitorres : :obj:`bool`, optional
            Monitor that residual is decreasing
    show : :obj:`bool`, optional
        Display iterations log
    itershow : :obj:`tuple`, optional
        Display set log for the first N1 steps, last N2 steps,
        and every N3 steps in between where N1, N2, N3 are the
        three element of the list.
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    preallocate : :obj:`bool`, optional
        Pre-allocate all variables used by the solver.
    Returns
    -------
    xinv : :obj:`pylops_mpi.DistributedArray`
        Inverted model
    niter : :obj:`int`
        Number of effective iterations
    cost : :obj:`numpy.ndarray`, optional
        History of cost (including regularization term)

    Raises
    ------
    NotImplementedError
        If ``threshkind`` is different from hard, soft, half.

    """
    callbacks = [
        CostNanInfCallback(),
    ]
    if rtol > 0.0:
        callbacks.append(CostToInitialCallback(rtol))
    if rtol1 > 0.0:
        callbacks.append(CostToDataCallback(rtol1))

    fistasolve = FISTA(
        Op,
        callbacks=callbacks,
    )
    if callback is not None:
        fistasolve.callback = callback
    x, iiter, cost = fistasolve.solve(
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
        itershow=itershow,
    )
    return x, iiter, cost
