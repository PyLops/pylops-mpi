from typing import Optional, Tuple, Callable

from pylops.utils import NDArray
from pylops_mpi import MPILinearOperator, DistributedArray
from pylops_mpi.optimization.cls_basic import CGLS


def cgls(
        Op: MPILinearOperator,
        y: DistributedArray,
        x0: Optional[DistributedArray] = None,
        niter: int = 10,
        damp: float = 0.0,
        tol: float = 1e-4,
        show: bool = False,
        itershow: Tuple[int, int, int] = (10, 10, 10),
        callback: Optional[Callable] = None,
) -> Tuple[DistributedArray, int, int, float, float, NDArray]:
    r"""Conjugate gradient least squares

    Solve an overdetermined system of equations given a MPILinearOperator ``Op`` and
    distributed data ``y`` using conjugate gradient iterations.

    Parameters
    ----------
    Op : :obj:`pylops_mpi.MPILinearOperator`
        MPI Linear Operator to invert of size :math:`[N \times M]`
    y : :obj:`pylops_mpi.DistributedArray`
        DistributedArray of size (N,)
    x0 : :obj:`pylops_mpi.DistributedArray`, optional
        Initial guess
    niter : :obj:`int`, optional
        Number of iterations
    damp : :obj:`float`, optional
        Damping coefficient
    tol : :obj:`float`, optional
        Tolerance on residual norm
    show : :obj:`bool`, optional
        Display iterations log
    itershow : :obj:`tuple`, optional
        Display set log for the first N1 steps, last N2 steps,
        and every N3 steps in between where N1, N2, N3 are the
        three element of the list.
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector

    Returns
    -------
    x : :obj:`pylops_mpi.DistributedArray`
        Estimated model of size (M, )
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

    Notes
    -----
    See :class:`pylops_mpi.optimization.cls_basic.CGLS`

    """
    cgsolve = CGLS(Op)
    if callback is not None:
        cgsolve.callback = callback
    x, istop, iiter, r1norm, r2norm, cost = cgsolve.solve(
        y=y, x0=x0, tol=tol, niter=niter, damp=damp, show=show, itershow=itershow
    )
    return x, istop, iiter, r1norm, r2norm, cost
