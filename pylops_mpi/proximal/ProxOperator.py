from mpi4py import MPI
from typing import Any, Callable

from pyproximal import ProxOperator
from pylops.utils.backend import get_module

from pylops_mpi import DistributedArray, Partition


_call_reduce_op = dict(
    Box=(MPI.LAND, all),
    L0=(MPI.SUM, sum),
    L1=(MPI.SUM, sum),
)


class MPIProxOperator:
    """MPI-enabled PyProximal Proximal Operator

    Common interface for applying (separable) proximal operators in a
    distributed fashion.

    In practice, this class provides methods to compute the norm, proximal
    operator and gradient between any :obj:`pyproximal.ProxOperator`
    (which must be the same across ranks) and a :class:`pylops_mpi.DistributedArray`.
    It internally handles the extraction of the local array from the distributed
    array and the creation of the output :class:`pylops_mpi.DistributedArray`.

    Parameters
    ----------
    prox : :obj:`pyproximal.ProxOperator`
        PyProximal Proximal Operator to wrap.
    
    """

    def __init__(
        self,
        prox: ProxOperator,
    ) -> None:
        # Check if prox is separable (by looking if is listed in
        # the mapping dictionary)
        prox_name = str(type(prox).__name__)
        if prox_name not in _call_reduce_op:
            raise NotImplementedError(
                f"{prox_name} is not a separable proximal "
                "operator, must be implemented directly...")
        self.proxop = prox
        self.hasgrad = prox.hasgrad

    def __repr__(self) -> str:
        if hasattr(self, "proxop"):
            return f"<{type(self).__name__} ({type(self.proxop).__name__})>"
        else:
            return f"<{type(self).__name__}>"
    
    def __call__(self, x: DistributedArray) -> DistributedArray:
        """Functional evaluation of the oprator.

        Modified version of pyproximal `__call__`. This method makes use
        of :class:`pylops_mpi.DistributedArray` to evaluate
        the functional of the operator in a distributed fashion.

        Parameters
        ----------
        x : :obj:`pylops_mpi.DistributedArray`
            A DistributedArray of global shape (N, ).

        Returns
        -------
        f : :obj:`bool` or :obj:`float` or :obj:`int`
            Function evaluation

        """
        if isinstance(x, DistributedArray):
            # Compute local function evaluation
            f = self.proxop(x.local_array)

            if x.partition == Partition.SCATTER:
                # Create receiver buffer
                ncp = get_module(x.engine)

                # Reduce local function evaluations into final evaluation
                reduce_op = _call_reduce_op[str(type(self.proxop).__name__)][0]
                recv_buf = x._allreduce_subcomm(x.sub_comm, x.base_comm_nccl,
                                                ncp.asarray(f),
                                                op=reduce_op,
                                                engine=x.engine)
                return recv_buf
            else:
                # For broadcasted arrays, simply return the local f
                return f
        else:  # StackedDistributedArray
            reduce_op = _call_reduce_op[str(type(self.proxop).__name__)][1]
            fs = [self(x[iarr]) for iarr in range(x.narrays)]
            f =  reduce_op(fs)
            return f
            
    def prox(self, x: DistributedArray, tau: float, **kwargs: Any) -> DistributedArray:
        """Proximal operator applied to a vector
        """
        if isinstance(x, DistributedArray):
            y = x.empty_like()
            y[:] = self.proxop.prox(x.local_array, tau)
        else:  # StackedDistributedArray
            y = x.empty_like()
            for iarr in range(x.narrays):
                y[iarr][:] = self.proxop.prox(x[iarr].local_array, tau)
        return y

    def proxdual(self, x: DistributedArray, tau: float, **kwargs: Any) -> DistributedArray:
        """Dual Proximal operator applied to a vector
        """
        y = DistributedArray(global_shape=x.global_shape,
                             base_comm=x.base_comm,
                             base_comm_nccl=x.base_comm_nccl,
                             partition=x.partition,
                             axis=x.axis,
                             local_shapes=x.local_shapes,
                             mask=x.mask,
                             engine=x.engine,
                             dtype=x.dtype)
        y[:] = self.proxop.proxdual(x.local_array, tau)

        return y
