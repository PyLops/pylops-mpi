import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Union

import numpy as np
from pylops.utils.backend import get_array_module, to_numpy
from pylops.utils.typing import NDArray

from pylops_mpi import DistributedArray
from pylops_mpi.proximal.ProxOperator import MPIProxOperator

if TYPE_CHECKING:
    from pylops_mpi.linearoperator import MPILinearOperator


def PrimalDual(
    proxf: MPIProxOperator,
    proxg: MPIProxOperator,
    A: "MPILinearOperator",
    x0: DistributedArray,
    y0: DistributedArray,
    tau: float | NDArray,
    mu: float | NDArray,
    z: DistributedArray | None = None,
    theta: float = 1.0,
    niter: int = 10,
    gfirst: bool = True,
    callback: Callable[[DistributedArray], None] | None = None,
    callbacky: bool = False,
    returny: bool = False,
    show: bool = False,
) -> Union[DistributedArray, tuple[DistributedArray, DistributedArray]]:
    r"""Primal-dual algorithm

    Solves the following (possibly) nonlinear minimization problem using
    the general version of the first-order primal-dual algorithm of [1]_:

    .. math::

        \min_{\mathbf{x} \in X} g(\mathbf{Ax}) + f(\mathbf{x}) +
        \mathbf{z}^T \mathbf{x}

    where :math:`\mathbf{A}` is a linear operator, :math:`f`
    and :math:`g` can be any convex functions that have a known proximal
    operator.

    This functional is effectively minimized by solving its equivalent
    primal-dual problem (primal in :math:`f`, dual in :math:`g`):

    .. math::

        \min_{\mathbf{x} \in X} \max_{\mathbf{y} \in Y}
        \mathbf{y}^T(\mathbf{Ax}) + \mathbf{z}^T \mathbf{x} +
        f(\mathbf{x}) - g^*(\mathbf{y})

    where :math:`\mathbf{y}` is the so-called dual variable.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    A : :obj:`pylops.LinearOperator`
        Linear operator of g
    x0 : :obj:`pylops_mpi.DistributedArray`
        Initial vector
    y0 : :obj:`pylops_mpi.DistributedArray`
        Initial auxiliary vector.
    tau : :obj:`float` or :obj:`np.ndarray`
        Stepsize of subgradient of :math:`f`. This can be constant
        or function of iterations (in the latter cases provided as np.ndarray)
    mu : :obj:`float` or :obj:`np.ndarray`
        Stepsize of subgradient of :math:`g^*`. This can be constant
        or function of iterations (in the latter cases provided as np.ndarray)
    z :  :obj:`pylops_mpi.DistributedArray`, optional
        Additional vector
    theta : :obj:`float`, optional
        Scalar between 0 and 1 that defines the update of the
        :math:`\bar{\mathbf{x}}` variable - note that ``theta=0`` is a
        special case that represents the semi-implicit classical Arrow-Hurwicz
        algorithm
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    callbacky : :obj:`bool`, optional
        Modify callback signature to (``callback(x, y)``) when ``callbacky=True``
    returny : :obj:`bool`, optional
        Return also ``y``
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`pylops_mpi.DistributedArray`
        Inverted model
    y : :obj:`pylops_mpi.DistributedArray`, optional
        Inverted second model, only returned if ``returny=True``

    Notes
    -----
    See :class:`pyproximal.optimization.primaldual.PrimalDual`

    """
    rank = x0.rank

    ncp = get_array_module(x0)

    # check if tau and mu are scalars or arrays
    fixedtau = fixedmu = False
    if isinstance(tau, (int, float)):
        tau = tau * ncp.ones(niter, dtype=np.float32)
        fixedtau = True
    if isinstance(mu, (int, float)):
        mu = mu * ncp.ones(niter, dtype=np.float32)
        fixedmu = True

    # initialize variables
    x = x0.copy()
    y = y0.copy() if y0 is not None else ncp.zeros(A.shape[0], dtype=x.dtype)
    xhat = x.copy()

    if show and rank == 0:
        tstart = time.time()
        print(
            "Primal-dual: min_x f(Ax) + x^T z + g(x)\n"
            "---------------------------------------------------------\n"
            "Proximal operator (f): %s\n"
            "Proximal operator (g): %s\n"
            "Linear operator (A): %s\n"
            "Additional vector (z): %s\n"
            "tau = %s\t\tmu = %s\ntheta = %.2f\t\tniter = %d\n"
            % (
                type(proxf),
                type(proxg),
                type(A),
                None if z is None else "vector",
                str(tau[0]) if fixedtau else "Variable",
                str(mu[0]) if fixedmu else "Variable",
                theta,
                niter,
            )
        )
        head = "   Itn       x[0]          f           g          z^x       J = f + g + z^x"
        print(head)
        sys.stdout.flush()

    # run iterations
    for iiter in range(niter):
        xold = x.copy()
        if gfirst:
            y = proxg.proxdual(y + mu[iiter] * A.matvec(xhat), mu[iiter])
            ATy = A.rmatvec(y)
            if z is not None:
                ATy += z
            x = proxf.prox(x - tau[iiter] * ATy, tau[iiter])
            xhat = x + theta * (x - xold)
        else:
            ATy = A.rmatvec(y)
            if z is not None:
                ATy += z
            x = proxf.prox(x - tau[iiter] * ATy, tau[iiter])
            xhat = x + theta * (x - xold)
            y = proxg.proxdual(y + mu[iiter] * A.matvec(xhat), mu[iiter])

        # run callback
        if callback is not None:
            if callbacky:
                callback(x, y)
            else:
                callback(x)
        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf, pg = proxf(x), proxg(A.matvec(x))
                pf = 0.0 if isinstance(pf, bool) else pf
                pg = 0.0 if isinstance(pg, bool) else pg
                zx = 0.0 if z is None else np.dot(z, x)
                if rank == 0:
                    msg = "%6g  %12.5e  %10.3e  %10.3e  %10.3e      %10.3e" % (
                        iiter + 1,
                        np.real(to_numpy(x[0])),
                        pf,
                        pg,
                        zx,
                        pf + pg + zx,
                    )
                    print(msg)
                    sys.stdout.flush()
    if show and rank == 0:
        print("\nTotal time (s) = %.2f" % (time.time() - tstart))
        print("---------------------------------------------------------\n")
        sys.stdout.flush()
    if not returny:
        return x
    else:
        return x, y
