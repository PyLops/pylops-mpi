from typing import Any

import numpy as np
from mpi4py import MPI
from pylops.utils.backend import get_module
from pylops.utils.typing import ShapeLike
from pyproximal.proximal import Simplex
from pyproximal.ProxOperator import _check_tau

from pylops_mpi import DistributedArray, Partition, StackedDistributedArray
from pylops_mpi.proximal import MPIProxOperator


class MPISimplex(MPIProxOperator):
    r"""Simplex proximal operator.

    Implement a distributed version of a Simplex:
    :math:`\Delta_n(r) = \{ \mathbf{x}:
    \sum_i x_i = r,\; x_i \geq 0 \}`.

    This operator can be applied to a set of vectors where each vector
    is created by stacking the elements at the same index of multiple
    :class:`pylops_mpi.DistributedArray` objects stored into a
    :class:`pylops_mpi.StackedDistributedArray`

    Parameters
    ----------
    dims : :obj:`tuple`, optional
        Dimensions of the local matrix onto which the input vector is reshaped.
        More specifically, ``dims[0]`` must match the number of
        :class:`pylops_mpi.DistributedArray` objects in the input
        :class:`pylops_mpi.StackedDistributedArray` and ``dims[1]`` is the
        local shape of each :class:`pylops_mpi.DistributedArray` object
    radius : :obj:`float`
        Radius
    maxiter : :obj:`int`, optional
        Maximum number of iterations used by bisection
    ftol : :obj:`float`, optional
        Function tolerance in bisection (only with ``engine='numba'``
        or ``engine='cuda'``)
    xtol : :obj:`float`, optional
        Solution absolute tolerance in bisection
    call : :obj:`bool`, optional
        Evalutate call method (``True``) or not (``False``)
    engine : :obj:`str`, optional
        Engine used for simplex computation (``numpy``, ``numba``or ``cuda``).

    Raises
    ------
    KeyError
        If ``engine`` is neither ``numpy`` nor ``numba`` nor ``cuda``

    Notes
    -----
    This is a distributed implementation of a Simplex repeatedly
    applied to a set of vectors which are defined as the columns
    of a matrix.

    The matrix is here represented as a
    :class:`pylops_mpi.StackedDistributedArray` with each row corresponding
    to a :class:`pylops_mpi.DistributedArray`. Each column is created by
    stacking the elements at the same index of multiple
    :class:`pylops_mpi.DistributedArray` objects.

    """

    def __init__(
        self,
        dims: ShapeLike,
        radius: float,
        maxiter: int = 100,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        call: bool = True,
        engine: str = "numpy",
    ) -> None:
        self.call = call
        self.simplex = Simplex(
            int(np.prod(dims)), radius, dims, 0, maxiter, ftol, xtol, call, engine
        )

    def __call__(self, x: StackedDistributedArray, tol: float = 1e-8) -> bool:
        if not self.call:
            return False
        # Create local input
        ncp = get_module(x.engine)
        xlocal = ncp.hstack([xi.local_array for xi in x])
        clocal = self.simplex(xlocal, tol)
        x0 = x[0]
        if x0.partition == Partition.SCATTER:
            # Constraint holds only if it holds on all ranks
            c = x0._allreduce_subcomm(
                x0.sub_comm,
                x0.base_comm_nccl,
                ncp.asarray(clocal),
                op=MPI.LAND,
                engine=x0.engine,
            )
            return bool(c.item())
        return bool(clocal)

    @_check_tau
    def prox(
        self, x: StackedDistributedArray, tau: float, **kwargs: Any
    ) -> DistributedArray:
        """Proximal operator applied to a vector"""
        ncp = get_module(x.engine)
        y = x.empty_like()
        xlocal = ncp.hstack([xi.local_array for xi in x])
        ylocal = self.simplex.prox(xlocal, tau).reshape(x.narrays, -1)
        for i in range(x.narrays):
            y[i].local_array[:] = ylocal[i]
        return y
