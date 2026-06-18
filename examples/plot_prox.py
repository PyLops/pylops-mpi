r"""
Proximal operators
==================

"""
import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt

import pylops
import pyproximal

import pylops_mpi

np.random.seed(42)
plt.close("all")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 10

# L1 norm
arr = pylops_mpi.DistributedArray(global_shape=n * size,
                                  partition=pylops_mpi.Partition.SCATTER)

arr[:] = rank * np.arange(n)

l1 = pyproximal.L1(sigma=2.0)
l1d = pylops_mpi.proximal.MPIProxOperator(l1)
f = l1d(arr)
prox = l1d.prox(arr, .1)
proxdlocal = prox.asarray()

dprox = l1d.proxdual(arr, .1)
dproxdlocal = dprox.asarray()

arrlocal = arr.asarray()
if rank == 0:
    flocal = l1(arrlocal)
    proxlocal = l1.prox(arrlocal, .1)
    dproxlocal = l1.proxdual(arrlocal, .1)
    print("||x||_1: ", f, flocal)
    print("prox_||x||_1: ", all(proxdlocal == proxlocal))
    print("proxd_||x||_1: ", all(dproxdlocal == dproxlocal))

# Box norm
arr = pylops_mpi.DistributedArray(global_shape=n * size,
                                  partition=pylops_mpi.Partition.SCATTER)

arr[:] = 3 * np.ones(n)
if rank == 0:
     arr[n//2] = 20

box = pyproximal.Box(lower=1., upper=5.)
boxd = pylops_mpi.proximal.MPIProxOperator(box)
f = boxd(arr)
prox = boxd.prox(arr, .1)
proxdlocal = prox.asarray()

dprox = boxd.proxdual(arr, .1)
dproxdlocal = dprox.asarray()

arrlocal = arr.asarray()
if rank == 0:
    flocal = box(arrlocal)
    proxlocal = box.prox(arrlocal, .1)
    dproxlocal = box.proxdual(arrlocal, .1)
    print("Box(x): ", f, flocal)
    print("prox_Box ", all(proxdlocal == proxlocal))
    print("proxd_Box ", all(dproxdlocal == dproxlocal))


# L2 norm ||x||_2^2
arr = pylops_mpi.DistributedArray(global_shape=n * size,
                                  partition=pylops_mpi.Partition.SCATTER)

arr[:] = rank * np.arange(n)

l2 = pyproximal.L2(sigma=2.0)
l2d = pylops_mpi.proximal.MPIL2(sigma=2.0)
f = l2d(arr)
prox = l2d.prox(arr, .1)
proxdlocal = prox.asarray()
grad = l2d.grad(arr)
graddlocal = grad.asarray()

arrlocal = arr.asarray()
if rank == 0:
    flocal = l2(arrlocal)
    proxlocal = l2.prox(arrlocal, .1)
    gradlocal = l2.grad(arrlocal)
    print("||x||_2^2: ", f, flocal)
    print("prox_||x||_2^2: ", all(proxdlocal == proxlocal))
    print("grad_||x||_2^2: ", all(graddlocal == gradlocal))

# L2 norm ||Op * x - d||_2^2
solver="cgls"
Op = pylops.FirstDerivative(n * size, sampling=0.001)
Opd = pylops_mpi.MPIFirstDerivative(n * size, sampling=0.001)
# Op = pylops.Diagonal(np.ones(n * size))
# Opd = pylops_mpi.MPIBlockDiag([pylops.Diagonal(np.ones(n)),])
  
b = pylops_mpi.DistributedArray(global_shape=n * size,
                                partition=pylops_mpi.Partition.SCATTER)

b[:] = rank * np.ones(n)
blocal = b.asarray()

x0 = arr.zeros_like()
x0local = x0.asarray()

l2 = pyproximal.L2(
    Op=Op, b=blocal, sigma=2.0,
    solver=solver, x0=x0local, kwargs_solver=dict(show=True))
l2d = pylops_mpi.proximal.MPIL2(
    Op=Opd, b=b, sigma=2.0,
    solver=solver, x0=x0, kwargs_solver=dict(show=True if rank==0 else False))
f = l2d(arr)
prox = l2d.prox(arr, .1)
proxdlocal = prox.asarray()
grad = l2d.grad(arr)
graddlocal = grad.asarray()

arrlocal = arr.asarray()
if rank == 0:
    flocal = l2(arrlocal)
    proxlocal = l2.prox(arrlocal, .1)
    gradlocal = l2.grad(arrlocal)
    print("||x||_2^2: ", f, flocal)
    print("prox_||x||_2^2: ", all(proxdlocal == proxlocal), np.linalg.norm(proxdlocal - proxlocal))
    print("grad_||x||_2^2: ", all(graddlocal == gradlocal))



# Proximal gradient
arr = pylops_mpi.DistributedArray(global_shape=n,
                                  partition=pylops_mpi.Partition.BROADCAST)
arr[:] = 0.0
arr[n//4] = 1.0
arr[n//2] = -0.5

Op = pylops.MatrixMult(np.random.normal(0, 1, (n-2, n,)))
Opd = pylops_mpi.MPILinearOperator(Op)

b = Opd @ arr
blocal = b.asarray()

l2d = pylops_mpi.proximal.MPIL2(
    Op=Opd, b=b, solver=solver, x0=arr.zeros_like())
l1 = pyproximal.L1(sigma=8e-1)
l1d = pylops_mpi.proximal.MPIProxOperator(l1)

arrpg = pylops_mpi.proximal.optimization.primal.ProximalGradient(
        l2d, l1d, x0=arr.zeros_like(), tau=1e-2, niter=400,
        show=True
    )
arrpgdlocal = arrpg.asarray()

arrlocal = arr.asarray()
if rank == 0:
    l2local = pyproximal.L2(
        Op=Op, b=blocal,
        solver=solver)
    l1local = pyproximal.L1(sigma=8e-1)

    arrpglocal = pyproximal.optimization.primal.ProximalGradient(
        l2local, l1local, x0=np.zeros(n), tau=1e-2, niter=400, show=True
    )  

    print('PG true', arrlocal)
    print('PG distr', arrpgdlocal)
    print('PG local', arrpglocal)


# ADMML2 with stacked operator
ny, nx = 40, 40
arrlocal = np.ones((ny, nx))
arrlocal[ny//2-5:ny//2+5, nx//2-5:nx//2+5] = 2
arr = pylops_mpi.DistributedArray(global_shape=ny * nx,
                                  partition=pylops_mpi.Partition.SCATTER)
arr[:] = arrlocal[ny//4 * rank: ny//4 * (rank +1)].flatten()

Op = pylops.Diagonal(np.ones(ny*nx))
Opd = pylops_mpi.MPIBlockDiag([pylops.Diagonal(np.ones(ny*nx//4)),])

b = Opd @ arr
blocal = b.asarray()

Iop = pylops.Identity(ny*nx)
Iopd = pylops_mpi.MPIBlockDiag([pylops.Identity(ny*nx//4),])

L = 8.0  # maxeig(Gop^H Gop)

l1 = pyproximal.L1(sigma=8e-1)
l1d = pylops_mpi.proximal.MPIProxOperator(l1)

x0distr = arr.zeros_like()
arradmm = pylops_mpi.proximal.optimization.primal.ADMML2(
        l1d, Opd, b, Iopd, x0=x0distr, tau=.99/L, niter=5,
        show=True, kwargs_solver=dict(niter=20),
    )[0]
arradmmdlocal = arradmm.asarray()

arrlocal = arr.asarray()
if rank == 0:

    l1local = pyproximal.L1(sigma=8e-1)

    arradmmlocal = pyproximal.optimization.primal.ADMML2(
        l1local, Op, blocal, Iop, x0=np.zeros(ny*nx), 
        tau=.99/L, niter=5, show=True, iter_lim=20,
    )[0]

    print('ADMML2 true', arrlocal)
    print('ADMML2 distr', arradmmdlocal)
    print('ADMML2 local', arradmmlocal)


# ADMML2 with stacked operator for A
ny, nx = 40, 40
arrlocal = np.ones((ny, nx))
arrlocal[ny//2-5:ny//2+5, nx//2-5:nx//2+5] = 2
arr = pylops_mpi.DistributedArray(global_shape=ny * nx,
                                  partition=pylops_mpi.Partition.SCATTER)
arr[:] = arrlocal[ny//4 * rank: ny//4 * (rank +1)].flatten()

Op = pylops.Diagonal(np.ones(ny*nx))
Opd = pylops_mpi.MPIBlockDiag([pylops.Diagonal(np.ones(ny*nx//4)),])

b = Opd @ arr
blocal = b.asarray()

Gopd = pylops_mpi.MPIGradient(
    dims=(ny, nx), sampling=1., edge=False, kind="forward")

L = 8.0  # maxeig(Gop^H Gop)

l1 = pyproximal.L1(sigma=8e-1)
l1d = pylops_mpi.proximal.MPIProxOperator(l1)

x0distr = arr.zeros_like()
arradmm = pylops_mpi.proximal.optimization.primal.ADMML2(
        l1d, Opd, b, Gopd, x0=x0distr, tau=.99/L, niter=5,
        show=True, kwargs_solver=dict(niter=5),
    )[0]
arradmmdlocal = arradmm.asarray()

arrlocal = arr.asarray()
if rank == 0:

    Gop = pylops.Gradient(
        dims=(ny, nx), sampling=1., edge=False, kind="forward",
        )
    l1local = pyproximal.L1(sigma=8e-1)

    arradmmlocal = pyproximal.optimization.primal.ADMML2(
        l1local, Op, blocal, Gop, x0=np.zeros(ny*nx), 
        tau=.99/L, niter=5, show=True, iter_lim=5,
    )[0]

    print('ADMML2 true', arrlocal)
    print('ADMML2 distr', arradmmdlocal)
    print('ADMML2 local', arradmmlocal)
    print(arradmmdlocal - arradmmlocal)

