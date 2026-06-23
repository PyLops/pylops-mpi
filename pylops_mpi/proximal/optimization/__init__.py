"""
Proximal Solvers using MPI
==========================

The subpackage proximal extends the pyproximal.optimization module
providing proximal solvers using MPI.


A list of proximal solvers:
    ProximalGradient                  Proximal Gradient

"""

from .primal import *


__all__ = [
    "ProximalGradient",
]