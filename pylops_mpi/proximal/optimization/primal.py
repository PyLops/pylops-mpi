import sys
import time
from collections.abc import Callable
from math import sqrt
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from pylops.optimization.leastsquares import regularized_inversion
from pylops.utils.backend import get_array_module, to_numpy
from pylops.utils.typing import NDArray

from pyproximal.proximal import L2

from pylops_mpi import DistributedArray
from pylops_mpi.proximal.ProxOperator import MPIProxOperator


if TYPE_CHECKING:
    from pylops_mpi.linearoperator import MPILinearOperator


def ProximalGradient(
    proxf: MPIProxOperator,
    proxg: MPIProxOperator,
    x0: DistributedArray,
    epsg: float | NDArray = 1.0,
    tau: float | None = None,
    # backtracking: bool = False,
    beta: float = 0.5,
    eta: float = 1.0,
    niter: int = 10,
    niterback: int = 100,
    acceleration: str | None = None,
    tol: float | None = None,
    callback: Callable[[NDArray], None] | None = None,
    show: bool = False,
) -> NDArray:
    r"""Proximal gradient (optionally accelerated)

    """
    rank = x0.rank
    
    # TODO: implement backtracking
    backtracking = False
    
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
            "tau = %s\tbacktrack = %s\tbeta = %10e\n"
            "epsg = %s\tniter = %d\ttol = %s\n"
            ""
            "niterback = %d\tacceleration = %s\n"
            % (
                type(proxf),
                type(proxg),
                str(tau),
                backtracking,
                beta,
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

    # if tau is None:
    #     backtracking = True
    #     tau = 1.0

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
        if not backtracking:
            if eta == 1.0:
                x = proxg.prox(y - tau * proxf.grad(y), epsg[iiter] * tau)
            else:
                x = x + eta * (
                    proxg.prox(x - tau * proxf.grad(x), epsg[iiter] * tau) - x
                )
        else:
            pass
            # x, tau = _backtracking(
            #     y, tau, proxf, proxg, epsg[iiter], beta=beta, niterback=niterback
            # )
            # if eta != 1.0:
            #     x = x + eta * (
            #         proxg.prox(x - tau * proxf.grad(x), epsg[iiter] * tau) - x
            #     )

        # update internal parameters for bilinear operator
        # if isinstance(proxf, BilinearOperator):
        #     proxf.updatexy(x)

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
    return x

