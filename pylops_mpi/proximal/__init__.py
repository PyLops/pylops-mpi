"""
Proximal Operators and Solvers using MPI
========================================

The subpackage proximal extends the pyproximal library providing
proximal operators and solvers using MPI.

A common interface for applying (separable) proximal operators in a
distributed fashion is provided by the MPIProxOperator operator.

A list of proximal operators present in pylops_mpi.proximal.proximal.

A list of proximal solvers present in pylops_mpi.proximal.optimization.primal.

"""

from .ProxOperator import *
from .proximal import *
from .optimization import *


__all__ = [
    "MPIProxOperator",
]
