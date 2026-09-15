"""
Proximal Solvers using MPI
==========================

The subpackage proximal extends the pyproximal.optimization module
providing proximal solvers using MPI.


A list of proximal solvers:
    ProximalGradient                  Proximal Gradient
    ADMML2                            ADMM with L2 misfit term
    PrimalDual                        Primal-Dual algorithm
"""

from .primal import *
from .primaldual import *


__all__ = [
    "ProximalGradient",
    "ADMML2",
    "PrimalDual",
]
