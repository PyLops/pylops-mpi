from typing import Optional, Dict, Any, Callable

from pylops.utils.typing import Union, Tuple, NDArray
from pylops.optimization.callback import CostNanInfCallback, CostToInitialCallback, CostToDataCallback

from pylops_mpi.DistributedArray import DistributedArray, StackedDistributedArray
from pylops_mpi.LinearOperator import MPILinearOperator
from pylops_mpi.optimization.cls_sparsity import ISTA


def ista(
    Op: Union[MPILinearOperator],
    y: Union[DistributedArray, StackedDistributedArray],
    x0: Optional[Union[DistributedArray, StackedDistributedArray]] = None,
    niter: int = 10,
    SOp: Optional[MPILinearOperator] = None,
    eps: float = 0.1,
    alpha: Optional[float] = None,
    eigsdict: Optional[Dict[str, Any]] = None,
    tol: float = 1e-10,
    rtol: float = 0.0,
    rtol1: float = 0.0,
    threshkind: str = "soft",
    perc: Optional[float] = None,
    decay: Optional[NDArray] = None,
    monitorres: bool = False,
    show: bool = False,
    itershow: Tuple[int, int, int] = (10, 10, 10),
    callback: Optional[Callable] = None,
    preallocate: bool = False,
) -> Tuple[Union[DistributedArray, StackedDistributedArray], int, NDArray]:
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
        perc=perc,
        decay=decay,
        monitorres=monitorres,
        show=show,
        itershow=itershow,
        preallocate=preallocate,
    )
    return x, iiter, cost