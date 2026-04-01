import time
import logging
import numpy as np
from typing import Optional, Dict, Any, Callable

from pylops.optimization.basesolver import Solver
from pylops.optimization.callback import _callback_stop
from pylops.utils import get_array_module, get_real_dtype
from pylops.utils.typing import NDArray, Union, Tuple

from pylops_mpi.DistributedArray import DistributedArray, StackedDistributedArray
from pylops_mpi.LinearOperator import MPILinearOperator
from pylops_mpi.optimization.eigs import power_iteration

logger = logging.getLogger(__name__)


def _hardthreshold(x: DistributedArray, thresh: float) -> DistributedArray:
    local = x.local_array
    local_out = local.copy()
    local_out[np.abs(local) <= np.sqrt(2 * thresh)] = 0.0

    dist = DistributedArray(
        global_shape=x.global_shape,
        partition=x.partition,
        base_comm=x.base_comm,
        base_comm_nccl=x.base_comm_nccl,
        dtype=x.dtype
    )
    dist[:] = local_out
    return dist


def _softthreshold(x: DistributedArray, thresh: float) -> DistributedArray:
    x_local = x.local_array
    if np.iscomplexobj(x_local):
        x1 = np.maximum(np.abs(x_local) - thresh, 0.0) * np.exp(1j * np.angle(x_local))
    else:
        x1 = np.maximum(np.abs(x_local) - thresh, 0.0) * np.sign(x_local)
    dist = DistributedArray(global_shape=x.global_shape, base_comm=x.base_comm, base_comm_nccl=x.base_comm_nccl,
                            partition=x.partition, dtype=x.dtype)
    dist[:] = x1
    return dist


def _halfthreshold(x: DistributedArray, thresh: float) -> DistributedArray:
    local = x.local_array
    ncp = get_array_module(local)
    arg = ncp.ones_like(local)
    mask = local != 0
    arg[mask] = (thresh / 8.0) * (ncp.abs(local[mask]) / 3.0) ** (-1.5)

    if ncp.iscomplexobj(arg):
        arg.real = ncp.clip(arg.real, -1.0, 1.0)
        arg.imag = ncp.clip(arg.imag, -1.0, 1.0)
    else:
        arg = ncp.clip(arg, -1.0, 1.0)

    phi = 2.0 / 3.0 * ncp.arccos(arg)

    local_out = 2.0 / 3.0 * local * (
            1.0 + ncp.cos(2.0 * np.pi / 3.0 - phi)
    )

    local_out[ncp.abs(local) <= (54.0 ** (1.0 / 3.0) / 4.0) * thresh ** (2.0 / 3.0)] = 0
    dist = DistributedArray(global_shape=x.global_shape, base_comm=x.base_comm, base_comm_nccl=x.base_comm_nccl,
                            partition=x.partition, dtype=x.dtype)
    dist[:] = local_out
    return dist


class ISTA(Solver):

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
                    backend="numpy",
                    **self.eigsdict,
                )[0]
            )
            self.alpha = 1.0 / maxeig
        self.thresh = eps * self.alpha * 0.5
        if x0 is None:
            x = DistributedArray(global_shape=self.Op.shape[1], dtype=self.Op.dtype)
            x[:] = 0
        else:
            x = x0.copy()
        if self.preallocate:
            self.res = DistributedArray(global_shape=y.global_shape, dtype=y.dtype)
            self.res = DistributedArray(global_shape=x.global_shape, dtype=x.dtype)
            self.x_unthesh = DistributedArray(global_shape=x.global_shape, dtype=x.dtype)
            self.xold = DistributedArray(global_shape=x.global_shape, dtype=x.dtype)
            if self.SOp is not None:
                self.SOpx_unthesh = DistributedArray(global_shape=self.SOp.shape[1], dtype=self.SOp.dtype)
                self.SOpx_unthesh[:] = 0
        if monitorres:
            self.normresold = np.inf
        self.t = 1.0
        self.cost = []
        self.iiter = 0
        if show:
            self._print_setup()
        return x

    def step(self, x: Union[DistributedArray, StackedDistributedArray], show: bool = False) -> Tuple[
        Union[DistributedArray, StackedDistributedArray], float]:
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
