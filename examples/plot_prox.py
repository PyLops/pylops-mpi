r"""
Proximal operators
==================

This example demonstrates the use of the :py:mod:`pylops_mpi.proximal`
module, and more specifically how to create and apply PyProximal operators
to distributed array.

"""
import numpy as np
import pylops
import pyproximal
from matplotlib import pyplot as plt
from mpi4py import MPI

import pylops_mpi

np.random.seed(42)
plt.close("all")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###############################################################################
# Let's start with so-called separable proximal operators. These are functionals
# whose proximal operator can be computed in a element-wise fashion. As such,
# no special implementation is required for the distributed counterpart of
# those operators. Instead, we can simply wrap the PyProximal operator into
# a :py:class:`pylops_mpi.proximal.MPIProxOperator`.
#
# We take the :py:class:`pyproximal.L1` norm as an example.

n = 10
arr = pylops_mpi.DistributedArray(
    global_shape=n * size, partition=pylops_mpi.Partition.SCATTER
)
arr[:] = (rank + 1) * np.arange(n)

l1 = pyproximal.proximal.L1(sigma=2.0)
l1d = pylops_mpi.proximal.MPIProxOperator(l1)

# Call
f = l1d(arr)

# Proximal
prox = l1d.prox(arr, 0.1)
proxdlocal = prox.asarray()

dprox = l1d.proxdual(arr, 0.1)
dproxdlocal = dprox.asarray()

arrlocal = arr.asarray()
if rank == 0:
    flocal = l1(arrlocal)
    proxlocal = l1.prox(arrlocal, 0.1)
    dproxlocal = l1.proxdual(arrlocal, 0.1)
    print("||x||_1: ", f, flocal)
    print("prox_||x||_1: ", all(proxdlocal == proxlocal))
    print("proxd_||x||_1: ", all(dproxdlocal == dproxlocal))

###############################################################################
# We repeat now the same with the :py:class:`pyproximal.Box` operator.

arr = pylops_mpi.DistributedArray(
    global_shape=n * size, partition=pylops_mpi.Partition.SCATTER
)

arr[:] = 3 * np.ones(n)
if rank == 0:
    arr[n // 2] = 20  # outside of the box

box = pyproximal.Box(lower=1.0, upper=5.0)
boxd = pylops_mpi.proximal.MPIProxOperator(box)

# Call
f = boxd(arr)

# Proximal
prox = boxd.prox(arr, 0.1)
proxdlocal = prox.asarray()

# Dual-Proximal
dprox = boxd.proxdual(arr, 0.1)
dproxdlocal = dprox.asarray()

arrlocal = arr.asarray()
if rank == 0:
    flocal = box(arrlocal)
    proxlocal = box.prox(arrlocal, 0.1)
    dproxlocal = box.proxdual(arrlocal, 0.1)
    print("Box(x): ", f, flocal)
    print("prox_Box ", all(proxdlocal == proxlocal))
    print("proxd_Box ", all(dproxdlocal == dproxlocal))

###############################################################################
# We move on now to a operator that is not separable and must be fully
# re-implemented in a distributed fashion, namely the
# :py:class:`pylops_mpi.proximal.proximal.MPIL2` norm.
#
# More precisely, when ``Op`` and ``b`` are passed to this operator,
# its proximal does call for the solution of a distributed inverse problem.
#
# However, let's start with the simplest case: :math:`||\mathbf{x}||_2^2`

arr = pylops_mpi.DistributedArray(
    global_shape=n * size, partition=pylops_mpi.Partition.SCATTER
)

arr[:] = (rank + 1) * np.arange(n)

l2 = pyproximal.L2(sigma=2.0)
l2d = pylops_mpi.proximal.MPIL2(sigma=2.0)

# Call
f = l2d(arr)

# Proximal
prox = l2d.prox(arr, 0.1)
proxdlocal = prox.asarray()

# Gradient
grad = l2d.grad(arr)
graddlocal = grad.asarray()

arrlocal = arr.asarray()
if rank == 0:
    flocal = l2(arrlocal)
    proxlocal = l2.prox(arrlocal, 0.1)
    gradlocal = l2.grad(arrlocal)
    print("||x||_2^2: ", f, flocal)
    print("prox_||x||_2^2: ", all(proxdlocal == proxlocal))
    print("grad_||x||_2^2: ", all(graddlocal == gradlocal))

###############################################################################
# Next we move onto the more general case, namely
# :math:`||\mathbf{Op} \mathbf{x} - \mathbf{b}||_2^2`

solver = "cgls"
Op = pylops.Diagonal(np.ones(n * size))
Opd = pylops_mpi.MPIBlockDiag(
    [
        pylops.Diagonal(np.ones(n)),
    ]
)

b = pylops_mpi.DistributedArray(
    global_shape=n * size, partition=pylops_mpi.Partition.SCATTER
)
b[:] = (rank + 1) * np.ones(n)
blocal = b.asarray()

x0 = arr.zeros_like()
x0local = x0.asarray()

l2 = pyproximal.L2(
    Op=Op, b=blocal, sigma=2.0, solver=solver, x0=x0local, kwargs_solver=dict(show=True)
)
l2d = pylops_mpi.proximal.MPIL2(
    Op=Opd,
    b=b,
    sigma=2.0,
    solver=solver,
    x0=x0,
    kwargs_solver=dict(show=True if rank == 0 else False),
)

# Call
f = l2d(arr)

# Proximal
prox = l2d.prox(arr, 0.1)
proxdlocal = prox.asarray()

# Gradient
grad = l2d.grad(arr)
graddlocal = grad.asarray()

arrlocal = arr.asarray()
if rank == 0:
    flocal = l2(arrlocal)
    proxlocal = l2.prox(arrlocal, 0.1)
    gradlocal = l2.grad(arrlocal)
    print("||Op . x - b||_2^2: ", f, flocal)
    print(
        "prox_||Op . x - b||_2^2 - norm diff=", np.linalg.norm(proxdlocal - proxlocal)
    )
    print("grad_||Op . x - b||_2^2: ", all(graddlocal == gradlocal))

###############################################################################
# We consider now another operator that is not separable and must be fully
# re-implemented in a distributed fashion, namely the
# :py:class:`pylops_mpi.proximal.proximal.MPIL21` norm.
#
# In this case the input is expected to be a stack of
# distributed arrays, namely a
# :py:class:`pylops_mpi.StackedDistributedArray` object with ``ndim``
# :py:class:`pylops_mpi.DistributedArray` objects.

# 2D input over which a gradient is computed
nx, ny = (16, 21)
arrlocal = np.random.normal(0, 1, (nx, ny))
arrlocal = arrlocal.flatten()
arr = pylops_mpi.DistributedArray.to_dist(x=arrlocal)

Gop = pylops_mpi.MPIGradient(dims=(nx, ny), dtype=np.float64)
grad = Gop @ arr

l21 = pyproximal.proximal.L21(ndim=2, sigma=2.0)
l21d = pylops_mpi.proximal.MPIL21(ndim=2, sigma=2.0)

# Call
f = l21d(grad)

# Proximal
prox = l21d.prox(grad, 0.1)
proxdlocal = prox.asarray()

dprox = l21d.proxdual(grad, 0.1)
dproxdlocal = dprox.asarray()

if rank == 0:
    Goplocal = pylops.Gradient(dims=(nx, ny), dtype=np.float64)
    gradlocal = Goplocal @ arrlocal
    flocal = l21(gradlocal)
    proxlocal = l21.prox(gradlocal, 0.1)
    dproxlocal = l21.proxdual(gradlocal, 0.1)
    print("||x||_2,1: ", f, flocal)
    print("prox_||x||_2,1: ", all(proxdlocal == proxlocal))
    print("proxd_||x||_2,1: ", all(dproxdlocal == dproxlocal))
