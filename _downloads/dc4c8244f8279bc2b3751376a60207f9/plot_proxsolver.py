r"""
Proximal solvers
================

This example demonstrates the use of the solvers in the
``pylops_mpi.proximal.optimization`` module.

"""
import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt

import pylops
import pyproximal

import pylops_mpi

plt.close("all")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.random.seed(rank)


###############################################################################
# Let's start with an example of sparsity promoting inversion using the
# :py:class:`pylops_mpi.proximal.optimization.primal.ProximalGradient` solver.
# Here for illustrative purposes, we consider a case where the model is 
# broadcasted whilst the data is scattered across ranks.

# Sparse input
n = 16
arr = pylops_mpi.DistributedArray(
    global_shape=n,
    partition=pylops_mpi.Partition.BROADCAST)
arr[:] = 0.0
arr[n // 4] = 1.0
arr[n // 2] = -0.5

# Operator and data
A = np.random.normal(0, 1, (n // size, n,))
Opd = pylops_mpi.MPIVStack([pylops.MatrixMult(A), ])

b = Opd @ arr
blocal = b.asarray()

# L2 prox
l2d = pylops_mpi.proximal.MPIL2(
    Op=Opd, b=b, x0=arr.zeros_like())

# L1 prox
l1 = pyproximal.L1(sigma=1e-1)
l1d = pylops_mpi.proximal.MPIProxOperator(l1)

# Distributed inversion
arrpg = pylops_mpi.proximal.optimization.primal.ProximalGradient(
    l2d, l1d, x0=arr.zeros_like(), tau=1e-2, niter=400,
    show=True,
)
arrpgdlocal = arrpg.asarray()

# Benchmark serial inversion
As = np.vstack(comm.allgather(A))
arrlocal = arr.asarray()
if rank == 0:
    Op = pylops.MatrixMult(As)
    l2local = pyproximal.L2(
        Op=Op, b=blocal)
    l1local = pyproximal.L1(sigma=1e-1)

    arrpglocal = pyproximal.optimization.primal.ProximalGradient(
        l2local, l1local, x0=np.zeros(n), tau=1e-2, niter=400, show=False
    )

    plt.figure(figsize=(12, 3))
    plt.plot(arrlocal, "k", label="True")
    plt.plot(arrpgdlocal, "b", label="Distr")
    plt.plot(arrpglocal, "--r", label="Local")
    plt.legend()
    plt.tight_layout()


###############################################################################
# Next we use the :py:class:`pylops_mpi.proximal.optimization.primal.ADMML2`
# solver for a similar problem. However we consider here a 2d array and impose
# blockiness in the solution. Once again, the model is broadcasted, 
# whilst the data is scattered.

# Input
ny, nx = 10 * size, 40
arrlocal = np.zeros((ny, nx))
arrlocal[ny // 2 - 5:ny // 2 + 5, nx // 2 - 5:nx // 2 + 5] = 2
arr = pylops_mpi.DistributedArray(
    global_shape=ny * nx,
    partition=pylops_mpi.Partition.BROADCAST
)
arr[:] = arrlocal.flatten()

# Operator and data
Op = pylops.VStack([pylops.Diagonal(np.ones(ny * nx)) for _ in range(size)])
Opd = pylops_mpi.MPIVStack([pylops.Diagonal(np.ones(ny * nx)),])

b = Opd @ arr
blocal = b.asarray()

# Regularizer
Gopd = pylops_mpi.MPILinearOperator(pylops.Gradient(
    dims=(ny, nx), sampling=1., edge=False, kind="forward"))

l1 = pyproximal.L1(sigma=2e0)
l1d = pylops_mpi.proximal.MPIProxOperator(l1)

# Distributed inversion
L = 8.0  # max eig of Gopd.H @ Gop
x0distr = arr.zeros_like()
arradmm = pylops_mpi.proximal.optimization.primal.ADMML2(
    l1d, Opd, b, Gopd, x0=x0distr, tau=.99 / L, niter=5,
    show=True, kwargs_solver=dict(niter=5),
)[0]
arradmmdlocal = arradmm.asarray()

# Benchmark serial inversion
arrlocal = arr.asarray()
if rank == 0:

    Gop = pylops.Gradient(
        dims=(ny, nx), sampling=1., edge=False, kind="forward",
    )
    l1local = pyproximal.L1(sigma=2e0)

    arradmmlocal = pyproximal.optimization.primal.ADMML2(
        l1local, Op, blocal, Gop, x0=np.zeros(ny * nx),
        tau=.99 / L, niter=5, show=False, iter_lim=5,
    )[0]

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    axs[0].imshow(arrlocal.reshape(10 * size, 40))
    axs[0].set_title("True")
    axs[0].axis("tight")
    axs[1].imshow(arradmmdlocal.reshape(10 * size, 40))
    axs[1].set_title("ADMML2 distr")
    axs[1].axis("tight")
    axs[2].imshow(arradmmlocal.reshape(10 * size, 40))
    axs[2].set_title("ADMML2 local")
    axs[2].axis("tight")
    fig.tight_layout()


###############################################################################
# And finally we repeat the same with a scattered model.

# Input
ny, nx = 10 * size, 40
arrlocal = np.zeros((ny, nx))
arrlocal[ny // 2 - 5:ny // 2 + 5, nx // 2 - 5:nx // 2 + 5] = 2
arr = pylops_mpi.DistributedArray(global_shape=ny * nx,
                                  partition=pylops_mpi.Partition.SCATTER)
arr[:] = arrlocal[ny // size * rank: ny // size * (rank + 1)].flatten()

# Operator and data
Op = pylops.Diagonal(np.ones(ny * nx))
Opd = pylops_mpi.MPIBlockDiag([pylops.Diagonal(np.ones((ny * nx) // size)),])

b = Opd @ arr
blocal = b.asarray()

# Regularizer
Gopd = pylops_mpi.MPIGradient(
    dims=(ny, nx), sampling=1., edge=False, kind="forward")

l1 = pyproximal.L1(sigma=2e0)
l1d = pylops_mpi.proximal.MPIProxOperator(l1)

# Distributed inversion
L = 8.0  # max eig of Gopd.H @ Gop
x0distr = arr.zeros_like()
arradmm = pylops_mpi.proximal.optimization.primal.ADMML2(
    l1d, Opd, b, Gopd, x0=x0distr, tau=.99 / L, niter=5,
    show=True, kwargs_solver=dict(niter=5),
)[0]
arradmmdlocal = arradmm.asarray()

# Benchmark serial inversion
arrlocal = arr.asarray()
if rank == 0:

    Gop = pylops.Gradient(
        dims=(ny, nx), sampling=1., edge=False, kind="forward"
    )
    l1local = pyproximal.L1(sigma=2e0)

    arradmmlocal = pyproximal.optimization.primal.ADMML2(
        l1local, Op, blocal, Gop, x0=np.zeros(ny * nx),
        tau=.99 / L, niter=5, show=False, iter_lim=5,
    )[0]

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    axs[0].imshow(arrlocal.reshape(10 * size, 40))
    axs[0].set_title("True")
    axs[0].axis("tight")
    axs[1].imshow(arradmmdlocal.reshape(10 * size, 40))
    axs[1].set_title("ADMML2 distr")
    axs[1].axis("tight")
    axs[2].imshow(arradmmlocal.reshape(10 * size, 40))
    axs[2].set_title("ADMML2 local")
    axs[2].axis("tight")
    fig.tight_layout()
