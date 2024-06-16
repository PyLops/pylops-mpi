"""
Wave Equation Operators using MPI
=====================================

The subpackage waveeqprocessing extends some of the wave equation processing
functionalities in pylops.waveeqprocessing providing forward and adjoint 
functionalities using MPI.

A list of operators present in pylops_mpi.waveeqprocessing :
    MPIMDC                      Multi-dimensional convolution.


"""

from .MDC import *

__all__ = [
    "MPIMDC",
]
