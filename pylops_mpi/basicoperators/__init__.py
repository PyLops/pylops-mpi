"""
Basic Linear Operators using MPI
================================

The subpackage basicoperators extends some of the basic linear algebra
operations provided by numpy providing forward and adjoint functionalities
using MPI.

A list of operators present in pylops_mpi.basicoperators :
    MPIBlockDiag                      Block Diagonal Operator

"""

from .BlockDiag import *

__all__ = [
    "MPIBlockDiag"
]
