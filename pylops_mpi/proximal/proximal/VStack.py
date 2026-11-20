from typing import Any

from pyproximal.ProxOperator import _check_tau

from pylops_mpi import StackedDistributedArray
from pylops_mpi.proximal import MPIProxOperator


class MPIProxStackedVStack(MPIProxOperator):
    r"""Vertical stacking operator.

    Implement a distributed version of :class:`pyproximal.VStack` operating on a
    :class:`pylops_mpi.StackedDistributedArray`. It effectively allows stacking
    a set of N distributed proximal operators vertically, each operating on a
    :class:`pylops_mpi.DistributedArray`. The overall proximal operator is
    computed as the stack of the output of each distributed proximal operator.

    Parameters
    ----------
    ops : :obj:`tuple`, optional
        Distributed proximal operators to be stacked

    """

    def __init__(
        self,
        ops: list[MPIProxOperator],
    ) -> None:
        self.ops = ops

    def __call__(self, x: StackedDistributedArray) -> float:
        f = 0
        for op, xd in zip(self.ops, x, strict=True):
            f += op(xd)
        return float(f)

    @_check_tau
    def prox(
        self, x: StackedDistributedArray, tau: float, **kwargs: Any
    ) -> StackedDistributedArray:
        y = [op.prox(xd, tau) for op, xd in zip(self.ops, x, strict=True)]
        y = StackedDistributedArray(y, base_comm=y[0].base_comm)
        return y
