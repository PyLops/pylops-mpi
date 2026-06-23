"""
Proximal Operators using MPI
============================

The subpackage proximal extends the pyproximal.proximal module providing
custom (non-separable) proximal operator using MPI, which cannot be
directly implemented by wrapping a PyProximal operator in
pylops_mpi.proximal.MPIProxOperator.


A list of proximal operators:
    MPIL2                             L2 Norm

"""

from .L2 import *


__all__ = [
    "MPIL2",
]