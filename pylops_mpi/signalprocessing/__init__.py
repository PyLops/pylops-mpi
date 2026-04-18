"""
Signal Processing Operators using MPI
=====================================

The subpackage signalprocessing extends some of the signal processing
functionalities in pylops.signalprocessing providing forward and adjoint
functionalities using MPI.

A list of operators present in pylops_mpi.signalprocessing :
    MPIFredholm1                      Fredholm integral of first kind.
    MPINonStationaryConvolve1D        1D non-stationary convolution operator.


"""

from .Fredholm1 import *
from .NonStatConvolve1d import *


__all__ = [
    "MPIFredholm1",
    "MPINonStationaryConvolve1D",
]
