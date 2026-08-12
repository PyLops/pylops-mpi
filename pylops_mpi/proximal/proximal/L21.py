from typing import Any

import numpy as np
from pylops.utils.backend import get_module, to_numpy
from pyproximal.ProxOperator import _check_tau

from pylops_mpi import DistributedArray, StackedDistributedArray
from pylops_mpi.proximal import MPIProxOperator


class MPIL21(MPIProxOperator):
    r""":math:`L_{2,1}` proximal operator.

    Implement a distributed version of the :math:`L_{2,1}` matrix norm.

    Parameters
    ----------
    ndim : :obj:`int`
        Number of dimensions :math:`N_{dim}`. Used to reshape the input array
        in a matrix of size :math:`N_{dim} \times N'_{x}` where
        :math:`N'_x = \frac{N_x}{N_{dim}}`. Note that the input
        vector ``x`` must be a :py:class:`pylops_mpi.StackedDistributedArray`
        that contains ``ndim`` :py:class:`pylops_mpi.DistributedArray`.
    sigma : :obj:`float`, optional
        Multiplicative coefficient of :math:`L_{2,1}` norm

    Notes
    -----
    This is a distributed implementation of the :math:`L_{2,1}` norm.

    XXXX

    """

    def __init__(
        self,
        ndim: int,
        sigma: float = 1.0,
    ) -> None:
        self.ndim = ndim
        self.sigma = sigma
        self.hasgrad = False

    def _check_dims(self, x: StackedDistributedArray) -> None:
        # Check that number of DistributedArray in x matches
        # with expected ndim
        if x.narrays != self.ndim:
            raise ValueError(f"Expected {self.ndim} DistributedArray, got {x.narrays}")

    def sum_squared(self, x: StackedDistributedArray) -> DistributedArray:
        # Square root of the sum of squared distributed arrays. Note that
        # the modulus is taken prior to summation, such that the returned
        # DistributedArray is always real-valued (also for complex-valued
        # inputs)
        ncp = get_module(x.engine)
        x0 = x.distarrays[0]
        sum2 = ncp.abs(x0.local_array) ** 2
        for iarr in range(1, self.ndim):
            sum2 += ncp.abs(x.distarrays[iarr].local_array) ** 2
        return DistributedArray(
            global_shape=x0.global_shape,
            base_comm=x0.base_comm,
            base_comm_nccl=x0.base_comm_nccl,
            partition=x0.partition,
            axis=x0.axis,
            local_array=ncp.sqrt(sum2),
            local_shapes=x0.local_shapes,
            mask=x0.mask,
            engine=x0.engine,
        )

    def __call__(self, x: StackedDistributedArray) -> float:
        # Check input
        self._check_dims(x)
        # Sum squared distributed arrays
        distrsum2 = self.sum_squared(x)
        # Compute norm
        f = self.sigma * distrsum2.norm(ord=1)
        return float(to_numpy(f.item()))

    @_check_tau
    def prox(self, x: DistributedArray, tau: float, **kwargs: Any) -> DistributedArray:
        """Proximal operator applied to a vector"""
        # Check input
        self._check_dims(x)

        # Sum squared distributed arrays
        distrsum2 = self.sum_squared(x)

        # Compute prox for each distributed array
        distrstacked = StackedDistributedArray(
            [x[iarr].empty_like() for iarr in range(self.ndim)]
        )
        for iarr in range(self.ndim):
            distrstacked[iarr].local_array[:] = (
                1
                - (tau * self.sigma)
                / np.maximum(distrsum2.local_array[:], tau * self.sigma)
            ) * x[iarr].local_array[:]
        return distrstacked

    @_check_tau
    def proxdual(
        self, x: DistributedArray, tau: float, **kwargs: Any
    ) -> DistributedArray:
        """Dual Proximal operator applied to a vector"""
        # Check input
        self._check_dims(x)

        # Sum squared distributed arrays
        distrsum2 = self.sum_squared(x)

        # Compute prox for each distributed array
        distrstacked = StackedDistributedArray(
            [x[iarr].empty_like() for iarr in range(self.ndim)]
        )
        for iarr in range(self.ndim):
            distrstacked[iarr].local_array[:] = (
                self.sigma
                * x[iarr].local_array[:]
                / np.maximum(distrsum2.local_array[:], self.sigma)
            )
        return distrstacked
