"""
Derivatives
===========
This example demonstrates how to use pylops-mpi's derivative operators, namely
:py:class:`pylops_mpi.basicoperators.MPIFirstDerivative` and
:py:class:`pylops_mpi.basicoperators.MPISecondDerivative`.

We will be focusing here on the case where the input array :math:`x` is assumed to be
an n-dimensional :py:class:`pylops_mpi.DistributedArray` and the derivative is
applied over the first axis (``axis=0``). Since the array is distributed
over multiple processes, the derivative operators must take care of applying
the derivatives across the edges using the information from the previous/next
processes, using the so-called ghost cells.

Derivative operators are commonly used when solving inverse problems within
regularization terms aimed at enforcing smooth solutions

"""
from matplotlib import pyplot as plt
import numpy as np
from mpi4py import MPI

import pylops_mpi

plt.close("all")
np.random.seed(42)

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

###############################################################################
# Let’s start by applying the first derivative on a :py:class:`pylops_mpi.DistributedArray`
# in the first direction(i.e. along axis=0) using the
# :py:class:`pylops_mpi.basicoperators.MPIFirstDerivative` operator.
nx, ny = 11, 21
x = np.zeros((nx, ny))
x[nx // 2, ny // 2] = 1.0

Fop = pylops_mpi.MPIFirstDerivative((nx, ny), dtype=np.float64)
x_dist = pylops_mpi.DistributedArray.to_dist(x=x.flatten())
y_dist = Fop @ x_dist
y = y_dist.asarray().reshape((nx, ny))

if rank == 0:
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    fig.suptitle(
        "First Derivative in 1st direction", fontsize=12, fontweight="bold", y=0.95
    )
    im = axs[0].imshow(x, interpolation="nearest", cmap="rainbow")
    axs[0].axis("tight")
    axs[0].set_title("x")
    plt.colorbar(im, ax=axs[0])
    im = axs[1].imshow(y, interpolation="nearest", cmap="rainbow")
    axs[1].axis("tight")
    axs[1].set_title("y")
    plt.colorbar(im, ax=axs[1])
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

###############################################################################
# We can now do the same for the second derivative using the
# :py:class:`pylops_mpi.basicoperators.MPISecondDerivative` operator.
nx, ny = 11, 21
x = np.zeros((nx, ny))
x[nx // 2, ny // 2] = 1.0

Sop = pylops_mpi.MPISecondDerivative(dims=(nx, ny), dtype=np.float64)
x_dist = pylops_mpi.DistributedArray.to_dist(x=x.flatten())
y_dist = Sop @ x_dist
y = y_dist.asarray().reshape((nx, ny))

if rank == 0:
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    fig.suptitle(
        "Second Derivative in 1st direction", fontsize=12, fontweight="bold", y=0.95
    )
    im = axs[0].imshow(x, interpolation="nearest", cmap="rainbow")
    axs[0].axis("tight")
    axs[0].set_title("x")
    plt.colorbar(im, ax=axs[0])
    im = axs[1].imshow(y, interpolation="nearest", cmap="rainbow")
    axs[1].axis("tight")
    axs[1].set_title("y")
    plt.colorbar(im, ax=axs[1])
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
