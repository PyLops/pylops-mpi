r"""
CGLS Solver
===========

This example demonstrates the utilization of :py:func:`pylops_mpi.optimization.basic.cgls` solver.
Our solver uses the :py:class:`pylops_mpi.DistributedArray` to reduce the following cost function
in a distributed fashion :

.. math::
        J = \| \mathbf{y} -  \mathbf{Ax} \|_2^2 + \epsilon \| \mathbf{x} \|_2^2

"""
import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt

import pylops

import pylops_mpi

np.random.seed(42)
plt.close("all")
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

###############################################################################
# Let's define a matrix with dimensions ``N`` and ``M`` and populate it with
# random numbers. Then, we will input this matrix in a
# :py:class:`pylops_mpi.basicoperators.MPIBlockDiag`.
N, M = 20, 15
Mop = pylops.MatrixMult(A=np.random.normal(0, 1, (N, M)))
BDiag = pylops_mpi.MPIBlockDiag(ops=[Mop, ], dtype=np.float128)

###############################################################################
# By applying the :py:class:`pylops_mpi.basicoperators.MPIBlockDiag` operator,
# we perform distributed matrix-vector multiplication.
x = pylops_mpi.DistributedArray(size * M, dtype=np.float128)
x[:] = np.ones(M)
y = BDiag @ x

###############################################################################
# We now utilize the cgls solver to obtain the inverse of the ``MPIBlockDiag``.
# In the case of MPIBlockDiag, each operator is responsible for performing
# an inversion operation iteratively at a specific rank. The resulting inversions
# are then obtained in a :py:class:`pylops_mpi.DistributedArray`. To obtain the
# overall inversion of the entire MPIBlockDiag, you can utilize the ``asarray()``
# function of the DistributedArray as shown below.

# Set initial guess `x0` to zeroes
x0 = pylops_mpi.DistributedArray(BDiag.shape[1], dtype=np.float128)
x0[:] = 0
xinv, istop, niter, r1norm, r2norm, cost = pylops_mpi.cgls(BDiag, y, x0=x0, niter=15, tol=1e-10, show=True)
xinv_array = xinv.asarray()

if rank == 0:
    print(f"CGLS Solution xinv={xinv_array}")
    # Visualize
    plt.figure(figsize=(18, 5))
    plt.plot(cost, lw=2, label="CGLS")
    plt.title("Cost Function")
    plt.legend()
    plt.tight_layout()
