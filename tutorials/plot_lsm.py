r"""
Least-squares Migration
=======================


"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI

from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.lsm import LSM

import pylops_mpi

np.random.seed(42)
plt.close("all")
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

# Velocity Model
nx, nz = 81, 60
dx, dz = 4, 4
x, z = np.arange(nx) * dx, np.arange(nz) * dz
v0 = 1000  # initial velocity
kv = 0.0  # gradient
vel = np.outer(np.ones(nx), v0 + kv * z)

# Reflectivity Model
refl = np.zeros((nx, nz))
refl[:, 30] = -1
refl[:, 50] = 0.5

# Receivers
nr = 11
rx = np.linspace(10 * dx, (nx - 10) * dx, nr)
rz = 20 * np.ones(nr)
recs = np.vstack((rx, rz))

# Sources
ns = 3
nstot = MPI.COMM_WORLD.allreduce(ns, MPI.SUM)
sxtot = np.linspace(dx * 10, (nx - 10) * dx, nstot)
sx = sxtot[rank * ns: (rank + 1) * ns]
print(sx)
sz = 10 * np.ones(ns)
sources = np.vstack((sx, sz))
ds = sources[0, 1] - sources[0, 0]

nt = 651
dt = 0.004
t = np.arange(nt) * dt
wav, wavt, wavc = ricker(t[:41], f0=20)

lsm = LSM(
    z,
    x,
    t,
    sources,
    recs,
    v0,
    wav,
    wavc,
    mode="analytic",
    engine="numba",
)

VStack = pylops_mpi.MPIVStack(ops=[lsm.Demop, ])
refl_dist = pylops_mpi.DistributedArray(global_shape=nx * nz, partition=pylops_mpi.Partition.BROADCAST)
refl_dist[:] = refl.flatten()
d_dist = VStack @ refl_dist
d = d_dist.asarray().reshape((nstot, nr, nt))

# Adjoint
madj = VStack.H @ d_dist
d_adj_dist = VStack @ madj
d_adj = d_adj_dist.asarray().reshape((nstot, nr, nt))

# Inverse
print(VStack)
print(d_dist)
x0 = pylops_mpi.DistributedArray(global_shape=4860, partition=pylops_mpi.Partition.BROADCAST)
x0[:] = 0
minv = pylops_mpi.cgls(VStack, d_dist, x0, niter=10, show=False)[0]
# d_inv_dist = VStack @ minv
# d_inv = d_inv_dist.asarray().reshape(ns, nr, nt)


if rank == 0:
    fig, axs = plt.subplots(1, 4, figsize=(10, 4))
    axs[0].imshow(d[0, :, :300].T, cmap="gray", vmin=-d.max(), vmax=d.max())
    axs[0].set_title(r"$d$")
    axs[0].axis("tight")
    axs[1].imshow(d_adj[0, :, :300].T, cmap="gray", vmin=-d_adj.max(), vmax=d_adj.max())
    axs[1].set_title(r"$d_{adj}$")
    axs[1].axis("tight")
    # axs[2].imshow(d_inv[0, :, :300].T, cmap="gray", vmin=-d.max(), vmax=d.max())
    # axs[2].set_title(r"$d_{inv}$")
    # axs[2].axis("tight")
    plt.show()
