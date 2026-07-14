import sys
import time
from collections.abc import Callable
from math import sqrt
from typing import TYPE_CHECKING, Any

import numpy as np
from pylops.utils.backend import to_numpy
from pylops.utils.typing import NDArray

from pyproximal.optimization.primal import _x0z0_init

from pylops_mpi import DistributedArray, StackedDistributedArray
from pylops_mpi.basicoperators import MPIStackedVStack
from pylops_mpi.optimization.basic import cgls
from pylops_mpi.proximal.ProxOperator import MPIProxOperator

if TYPE_CHECKING:
    from pylops_mpi.linearoperator import MPILinearOperator


def ProximalGradient(
    proxf: MPIProxOperator,
    proxg: MPIProxOperator,
    x0: DistributedArray,
    epsg: float | NDArray = 1.0,
    tau: float | None = None,
    eta: float = 1.0,
    niter: int = 10,
    niterback: int = 100,
    acceleration: str | None = None,
    tol: float | None = None,
    callback: Callable[[DistributedArray], None] | None = None,
    show: bool = False,
) -> DistributedArray:
    r"""Proximal gradient (optionally accelerated)

    Solves the following minimization problem using (Accelerated) Proximal
    gradient algorithm:

    .. math::

        \mathbf{x} = \arg\,min_\mathbf{x} f(\mathbf{x}) + \epsilon g(\mathbf{x})

    where :math:`f(\mathbf{x})` is a smooth convex function with a uniquely
    defined gradient and :math:`g(\mathbf{x})` is any convex function that
    has a known proximal operator. Both ``f`` and ``g`` must be of
    :class:`pylops_mpi.proximal.MPIProxOperator` kind.

    Parameters
    ----------
    proxf : :obj:`pylops_mpi.proximal.MPIProxOperator`
        Proximal operator of f function (must have ``grad`` implemented)
    proxg : :obj:`pylops_mpi.proximal.MPIProxOperator`
        Proximal operator of g function
    x0 : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
        Initial vector
    epsg : :obj:`float` or :obj:`numpy.ndarray`, optional
        Scaling factor of g function. Can be a scalar
        for iteration-independent scaling or a a 1d vector for
        iteration-dependent scaling
    tau : :obj:`float`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\nabla f`.
    eta : :obj:`float`, optional
        Relaxation parameter (must be between 0 and 1, 0 excluded).
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    niterback : :obj:`int`, optional
        Max number of iterations of backtracking
    acceleration : :obj:`str`, optional
        Acceleration (``None``, ``vandenberghe`` or ``fista``)
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached or the other tolerance
        criterion is met
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
        Inverted model

    Notes
    -----
    See :class:`pyproximal.optimization.primal.ProximalGradient`

    """
    rank = x0.rank

    # check if epgs is a vector
    epsg = np.asarray(epsg, dtype=float)
    if epsg.size == 1:
        epsg = epsg * np.ones(niter)
        epsg_print = str(epsg[0])
    else:
        epsg_print = "Multi"

    if acceleration not in [None, "None", "vandenberghe", "fista"]:
        msg = "Acceleration should be None, vandenberghe or fista"
        raise NotImplementedError(msg)
    if show and rank == 0:
        tstart = time.time()
        print(
            "Accelerated Proximal Gradient\n"
            "---------------------------------------------------------\n"
            "Proximal operator (f): %s\n"
            "Proximal operator (g): %s\n"
            "tau = %s\tepsg = %s\n"
            "niter = %d\ttol = %s\n"
            ""
            "niterback = %d\tacceleration = %s\n"
            % (
                proxf,
                proxg,
                str(tau),
                epsg_print,
                niter,
                str(tol),
                niterback,
                acceleration,
            )
        )
        head = "   Itn       x[0]          f           g       J=f+eps*g       tau"
        print(head)
        sys.stdout.flush()

    # initialize model
    t = 1.0
    x = x0.copy()
    y = x.copy()
    pfg = np.inf
    tolbreak = False

    # iterate
    for iiter in range(niter):
        xold = x.copy()

        # proximal step
        if eta == 1.0:
            x = proxg.prox(y - tau * proxf.grad(y), epsg[iiter] * tau)
        else:
            x = x + eta * (
                proxg.prox(x - tau * proxf.grad(x), epsg[iiter] * tau) - x
            )

        # update y
        if acceleration == "vandenberghe":
            omega = iiter / (iiter + 3)
        elif acceleration == "fista":
            told = t
            t = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            omega = (told - 1.0) / t
        else:
            omega = 0
        y = x + omega * (x - xold)

        # run callback
        if callback is not None:
            callback(x)

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if tol is not None:
            pfgold = pfg
            pf, pg = proxf(x), proxg(x)
            pfg = pf + np.sum(epsg[iiter] * pg)
            if np.abs(1.0 - pfg / pfgold) < tol:
                tolbreak = True

        # show iteration logger
        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                if tol is None:
                    pf, pg = proxf(x), proxg(x)
                    pfg = pf + np.sum(epsg[iiter] * pg)
                if rank == 0:
                    msg = "%6g  %12.5e  %10.3e  %10.3e  %10.3e  %10.3e" % (
                        iiter + 1,
                        (
                            np.real(to_numpy(x[0]))
                            if x.ndim == 1
                            else np.real(to_numpy(x[0, 0]))
                        ),
                        pf,
                        pg,
                        pfg,
                        tau,
                    )
                    print(msg)
                    sys.stdout.flush()

        # break if tolerance condition is met
        if tolbreak:
            break

    if show and rank == 0:
        print("\nTotal time (s) = %.2f" % (time.time() - tstart))
        print("---------------------------------------------------------\n")
        sys.stdout.flush()
    return x


def ADMML2(
    proxg: MPIProxOperator,
    Op: "MPILinearOperator",
    b: DistributedArray,
    A: "MPILinearOperator",
    x0: DistributedArray,
    tau: float,
    niter: int = 10,
    z0: DistributedArray | None = None,
    gfirst: bool = False,
    callback: Callable[[DistributedArray], None] | None = None,
    show: bool = False,
    kwargs_solver: dict[str, Any] = {},
) -> tuple[DistributedArray, DistributedArray]:
    r"""Alternating Direction Method of Multipliers for L2 misfit term

    Solves the following minimization problem using Alternating Direction
    Method of Multipliers:

    .. math::

        \mathbf{x},\mathbf{z}  = \arg\,min_{\mathbf{x},\mathbf{z}}
        \frac{1}{2}||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2 + g(\mathbf{z}) \\
        s.t. \; \mathbf{Ax}=\mathbf{z}

    where :math:`g(\mathbf{z})` is any convex function that has a known proximal operator.

    Parameters
    ----------
    proxg : :obj:`pylops_mpi.proximal.MPIProxOperator`
        Proximal operator of g function
    Op : :obj:`pylops_mpi.MPILinearOperator` or :obj:`pylops_mpi.MPIStackedLinearOperator`
        Linear operator of data misfit term
    b : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
        Data
    A : :obj:`pylops_mpi.MPILinearOperator` or :obj:`pylops_mpi.MPIStackedLinearOperator`
        Linear operator of regularization term
    x0 : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
        Initial vector
    tau : :obj:`float`
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau \in (0, 1/\lambda_{max}(\mathbf{A}^H\mathbf{A})]`.
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    z0 : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
        Initial auxiliary vector. If ``None``, initialized to ``A @ x0``.
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log
    **kwargs_solver
        Arbitrary keyword arguments for :py:func:`pylops_mpi.optimization.basic.cgls` used
        to solve the x-update

    Returns
    -------
    x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
        Inverted model
    z : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
        Inverted second model

    Raises
    ------
    ValueError
        If both ``x0`` and ``z0`` are set to ``None`` or ``x0`` is set to None

    Notes
    -----
    See :class:`pyproximal.optimization.primal.ADMML2`

    """
    rank = x0.rank

    # initialize variables
    x, z = _x0z0_init(x0, z0, A, Opname="A")
    u = z.zeros_like()

    if show and rank == 0:
        tstart = time.time()
        print(
            "ADMM\n"
            "---------------------------------------------------------\n"
            "Proximal operator (g): %s\n"
            "tau = %10e\tniter = %d\n" % (proxg, tau, niter)
        )
        head = "   Itn       x[0]          f           g       J = f + g"
        print(head)
        sys.stdout.flush()

    # run iterations
    sqrttau = 1.0 / sqrt(tau)
    for iiter in range(niter):
        if gfirst:
            Ax = A @ x
            z = proxg.prox(Ax + u, tau)

            # solve augumented system
            Opreg = MPIStackedVStack([Op, sqrttau * A])
            breg = StackedDistributedArray([b, sqrttau * (z - u)])
            x = cgls(Opreg, breg, x0=x, **kwargs_solver)[0]
        else:
            # solve augumented system
            Opreg = MPIStackedVStack([Op, sqrttau * A])
            breg = StackedDistributedArray([b, sqrttau * (z - u)])
            x = cgls(Opreg, breg, x0=x, **kwargs_solver)[0]

            Ax = A @ x
            z = proxg.prox(Ax + u, tau)
        u = u + Ax - z

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf, pg = 0.5 * (Op @ x - b).norm() ** 2, proxg(Ax)
                if rank == 0:
                    msg = "%6g  %12.5e  %10.3e  %10.3e  %10.3e" % (
                        iiter + 1,
                        np.real(to_numpy(x[0])),
                        pf,
                        pg,
                        pf + pg,
                    )
                    print(msg)
                    sys.stdout.flush()
    if show and rank == 0:
        print("\nTotal time (s) = %.2f" % (time.time() - tstart))
        print("---------------------------------------------------------\n")
        sys.stdout.flush()
    return x, z
