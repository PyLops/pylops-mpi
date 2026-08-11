from typing import Any

from mpi4py import MPI
from pylops.utils.backend import get_module
from pyproximal import ProxOperator
from pyproximal.ProxOperator import _check_tau

from pylops_mpi import DistributedArray, Partition, StackedDistributedArray

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
                "operator, must be implemented directly..."
            )
        self.proxop = prox
        self.hasgrad = prox.hasgrad

    def __repr__(self) -> str:
        if hasattr(self, "proxop"):
            return f"<{type(self).__name__} ({type(self.proxop).__name__})>"
        else:
            return f"<{type(self).__name__}>"

    def __call__(
        self,
        x: DistributedArray | StackedDistributedArray,
    ) -> bool | float | int:
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

        def _as_scalar(value):
            """Convert NumPy/CuPy/Python scalar-like objects to a Python scalar."""
            # Ensure that a bool/int/float is returned
            if isinstance(value, (bool, int, float)):
                return value

            ncp = get_module(x.engine)
            if ncp.size(value) != 1:
                raise ValueError(
                    f"Expected scalar function evaluation, "
                    f"got object with shape {getattr(value, 'shape', None)}"
                )
            return value.item()

        if isinstance(x, DistributedArray):
            # Compute local function evaluation
            f = self.proxop(x.local_array)

            if x.partition == Partition.SCATTER:
                # Create receiver buffer
                ncp = get_module(x.engine)

                # Reduce local function evaluations into final evaluation
                reduce_op = _call_reduce_op[str(type(self.proxop).__name__)][0]
                recv_buf = x._allreduce_subcomm(
                    x.sub_comm,
                    x.base_comm_nccl,
                    ncp.asarray(f),
                    op=reduce_op,
                    engine=x.engine,
                )

                return _as_scalar(recv_buf)
            else:
                # For broadcasted arrays, simply return the local evaluation
                return _as_scalar(f)
        else:  # StackedDistributedArray
            reduce_op = _call_reduce_op[str(type(self.proxop).__name__)][1]
            fs = [self(x[iarr]) for iarr in range(x.narrays)]
            f = reduce_op(fs)
            return f

    @_check_tau
    def prox(
        self,
        x: DistributedArray | StackedDistributedArray,
        tau: float,
        **kwargs: Any,
    ) -> DistributedArray | StackedDistributedArray:
        """Proximal operator applied to a vector"""
        if isinstance(x, DistributedArray):
            y = x.empty_like()
            y[:] = self.proxop.prox(x.local_array, tau)
        else:  # StackedDistributedArray
            y = x.empty_like()
            for iarr in range(x.narrays):
                y[iarr][:] = self.proxop.prox(x[iarr].local_array, tau)
        return y

    @_check_tau
    def proxdual(
        self,
        x: DistributedArray | StackedDistributedArray,
        tau: float,
        **kwargs: Any,
    ) -> DistributedArray | StackedDistributedArray:
        """Dual Proximal operator applied to a vector"""
        y = DistributedArray(
            global_shape=x.global_shape,
            base_comm=x.base_comm,
            base_comm_nccl=x.base_comm_nccl,
            partition=x.partition,
            axis=x.axis,
            local_shapes=x.local_shapes,
            mask=x.mask,
            engine=x.engine,
            dtype=x.dtype,
        )
        y[:] = self.proxop.proxdual(x.local_array, tau)

        return y

    def precomposition(
        self,
        a: float,
        b: float | DistributedArray | StackedDistributedArray,
    ) -> "MPIProxOperator":
        r"""Precomposition

        Multiplies scalar ``a`` and adds scalar or vector ``b`` to
        ``x`` when evaluating the proximal function

        Parameters
        ----------
        a : :obj:`float`
            Multiplicative scalar
        b : :obj:`float` or obj:`pylops_mpi.DistributedArray` or obj:`pylops_mpi.StackedDistributedArray`
            Additive scalar (or vector)

        Notes
        -----
        The proximal operator of a function :math:`g= f(a \mathbf{x} + b)` is
        defined as:

        .. math::

            prox_{\tau g} (\mathbf{x}) = \frac{1}{a} (
            prox_{a^2 \tau f} (a \mathbf{x} + b) - b)

        """
        if isinstance(a, float) and isinstance(
            b, (float, DistributedArray, StackedDistributedArray)
        ):
            return _PrecompositionOperator(self, a, b)
        else:
            msg = "a must be of type float and b must be of type float, DistributedArray, or StackedDistributedArray"
            raise NotImplementedError(msg)


class _PrecompositionOperator(MPIProxOperator):
    def __init__(
        self,
        f: MPIProxOperator,
        a: float,
        b: float | DistributedArray | StackedDistributedArray,
    ) -> None:
        if not isinstance(a, float):
            msg = "Second input must be a float"
            raise ValueError(msg)
        if not isinstance(b, (float, DistributedArray, StackedDistributedArray)):
            msg = "Third input must be a float, DistributedArray, or StackedDistributedArray"
            raise ValueError(msg)
        self.f, self.a, self.b = f, a, b
        self.hasgrad = f.hasgrad

    def __call__(
        self, x: DistributedArray | StackedDistributedArray
    ) -> DistributedArray | StackedDistributedArray:
        return self.f(self.a * x + self.b)

    @_check_tau
    def prox(
        self, x: DistributedArray | StackedDistributedArray, tau: float, **kwargs: Any
    ) -> DistributedArray | StackedDistributedArray:
        return (1.0 / self.a) * (
            self.f.prox(self.a * x + self.b, (self.a**2) * tau) - self.b
        )

    def grad(
        self, x: DistributedArray | StackedDistributedArray
    ) -> DistributedArray | StackedDistributedArray:
        return self.a * self.f.grad(self.a * x + self.b)
