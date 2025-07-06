r"""
Least-squares Migration with NCCL
=================================
This tutorial is an extension of the :ref:`sphx_glr_tutorials_lsm.py`
tutorial where PyLops-MPI is run in multi-GPU setting with GPUs communicating
via NCCL.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
from mpi4py import MPI

from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.lsm import LSM

import pylops_mpi

###############################################################################
# NCCL communication can be easily initialized with
# :py:func:`pylops_mpi.utils._nccl.initialize_nccl_comm` operator.
# One can think of this as GPU-counterpart of :code:`MPI.COMM_WORLD`

np.random.seed(42)
plt.close("all")
nccl_comm = pylops_mpi.utils._nccl.initialize_nccl_comm()
rank = MPI.COMM_WORLD.Get_rank()

###############################################################################
# Let's start by defining all the parameters required by the
# :py:class:`pylops.waveeqprocessing.LSM` operator.
# Note that this section is exactly the same as the one in the MPI example
# as we will keep using MPI for transfering metadata (i.e., shapes, dims, etc.)

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
ns = 10
# Total number of sources at all ranks
nstot = MPI.COMM_WORLD.allreduce(ns, op=MPI.SUM)
sxtot = np.linspace(dx * 10, (nx - 10) * dx, nstot)
sx = sxtot[rank * ns: (rank + 1) * ns]
sztot = 10 * np.ones(nstot)
sz = 10 * np.ones(ns)
sources = np.vstack((sx, sz))
sources_tot = np.vstack((sxtot, sztot))

if rank == 0:
    plt.figure(figsize=(10, 5))
    im = plt.imshow(vel.T, cmap="summer", extent=(x[0], x[-1], z[-1], z[0]))
    plt.scatter(recs[0], recs[1], marker="v", s=150, c="b", edgecolors="k")
    plt.scatter(sources_tot[0], sources_tot[1], marker="*", s=150, c="r", edgecolors="k")
    cb = plt.colorbar(im)
    cb.set_label("[m/s]")
    plt.axis("tight")
    plt.xlabel("x [m]"), plt.ylabel("z [m]")
    plt.title("Velocity")
    plt.xlim(x[0], x[-1])
    plt.tight_layout()

    plt.figure(figsize=(10, 5))
    im = plt.imshow(refl.T, cmap="gray", extent=(x[0], x[-1], z[-1], z[0]))
    plt.scatter(recs[0], recs[1], marker="v", s=150, c="b", edgecolors="k")
    plt.scatter(sources_tot[0], sources_tot[1], marker="*", s=150, c="r", edgecolors="k")
    plt.colorbar(im)
    plt.axis("tight")
    plt.xlabel("x [m]"), plt.ylabel("z [m]")
    plt.title("Reflectivity")
    plt.xlim(x[0], x[-1])
    plt.tight_layout()

###############################################################################
# We create a :py:class:`pylops.waveeqprocessing.LSM` at each rank and then push them
# into a :py:class:`pylops_mpi.basicoperators.MPIVStack` to perform a matrix-vector
# product with the broadcasted reflectivity at every location on the subsurface.
# Also, we must pass `nccl_comm` to `refl` in order to use NCCL for communications.
# Noted that we allocate some arrays (wav, lsm.Demop.trav_srcs, and lsm.Demop.trav.recs)
# to GPU upfront. Because we want a fair performace comparison, we avoid having
# LSM internally copying arrays.

# Wavelet
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
    cp.asarray(wav.astype(np.float32)),
    wavc,
    mode="analytic",
    engine="cuda",
    dtype=np.float32
)
lsm.Demop.trav_srcs = cp.asarray(lsm.Demop.trav_srcs.astype(np.float32))
lsm.Demop.trav_recs = cp.asarray(lsm.Demop.trav_recs.astype(np.float32))

VStack = pylops_mpi.MPIVStack(ops=[lsm.Demop, ])
refl_dist = pylops_mpi.DistributedArray(global_shape=nx * nz,
                                        partition=pylops_mpi.Partition.BROADCAST,
                                        base_comm_nccl=nccl_comm,
                                        engine="cupy")
refl_dist[:] = cp.asarray(refl.flatten())
d_dist = VStack @ refl_dist
d = d_dist.asarray().reshape((nstot, nr, nt))

###############################################################################
# We calculate now the adjoint and model the data using the adjoint reflectivity
# as input.
madj_dist = VStack.H @ d_dist
madj = madj_dist.asarray().reshape((nx, nz))
d_adj_dist = VStack @ madj_dist
d_adj = d_adj_dist.asarray().reshape((nstot, nr, nt))

###############################################################################
# We calculate the inverse using the :py:func:`pylops_mpi.optimization.basic.cgls`
# solver. Here, we pass the `nccl_comm` to `x0` to use NCCL as a communicator.
# In this particular case, the local computation will be done in GPU.
# Collective communication calls will be carried through NCCL GPU-to-GPU.

# Inverse
# Initializing x0 to zeroes
x0 = pylops_mpi.DistributedArray(VStack.shape[1],
                                 partition=pylops_mpi.Partition.BROADCAST,
                                 base_comm_nccl=nccl_comm,
                                 engine="cupy")
x0[:] = 0
minv_dist = pylops_mpi.cgls(VStack, d_dist, x0=x0, niter=100, show=True)[0]
minv = minv_dist.asarray().reshape((nx, nz))
d_inv_dist = VStack @ minv_dist
d_inv = d_inv_dist.asarray().reshape(nstot, nr, nt)

##############################################################################
# Finally we visualize the results. Note that the array must be copied back
# to the CPU by calling the :code:`get()` method on the CuPy arrays.

if rank == 0:
    # Visualize
    fig1, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(refl.T, cmap="gray", vmin=-1, vmax=1)
    axs[0].axis("tight")
    axs[0].set_title(r"$m$")
    axs[1].imshow(madj.T.get(), cmap="gray", vmin=-madj.max(), vmax=madj.max())
    axs[1].set_title(r"$m_{adj}$")
    axs[1].axis("tight")
    axs[2].imshow(minv.T.get(), cmap="gray", vmin=-1, vmax=1)
    axs[2].axis("tight")
    axs[2].set_title(r"$m_{inv}$")
    plt.tight_layout()
    fig1.savefig("model.png")

    fig2, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(d[0, :, :300].T.get(), cmap="gray", vmin=-d.max(), vmax=d.max())
    axs[0].set_title(r"$d$")
    axs[0].axis("tight")
    axs[1].imshow(d_adj[0, :, :300].T.get(), cmap="gray", vmin=-d_adj.max(), vmax=d_adj.max())
    axs[1].set_title(r"$d_{adj}$")
    axs[1].axis("tight")
    axs[2].imshow(d_inv[0, :, :300].T.get(), cmap="gray", vmin=-d.max(), vmax=d.max())
    axs[2].set_title(r"$d_{inv}$")
    axs[2].axis("tight")
    fig2.savefig("data1.png")

    fig3, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(d[nstot // 2, :, :300].T.get(), cmap="gray", vmin=-d.max(), vmax=d.max())
    axs[0].set_title(r"$d$")
    axs[0].axis("tight")
    axs[1].imshow(d_adj[nstot // 2, :, :300].T.get(), cmap="gray", vmin=-d_adj.max(), vmax=d_adj.max())
    axs[1].set_title(r"$d_{adj}$")
    axs[1].axis("tight")
    axs[2].imshow(d_inv[nstot // 2, :, :300].T.get(), cmap="gray", vmin=-d.max(), vmax=d.max())
    axs[2].set_title(r"$d_{inv}$")
    axs[2].axis("tight")
    plt.tight_layout()
    fig3.savefig("data2.png")
