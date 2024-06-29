"""
Signal Processing Operators using MPI
=====================================

The subpackage signalprocessing extends some of the signal processing
functionalities in pylops.signalprocessing providing forward and adjoint
functionalities using MPI.

A list of operators present in pylops_mpi.signalprocessing :
    MPIFredholm1                      Fredholm integral of first kind.


"""

from .Fredholm1 import *

__all__ = [
    "MPIFredholm1",
]
