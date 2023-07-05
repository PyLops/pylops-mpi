"""
Post Stack Inversion - 3D
=========================
This illustration demonstrates the implementation of a distributed 3D Post-stack inversion. It involves
modelling a 3-D synthetic post-stack seismic data from a profile of the subsurface acoustic impedence.
"""

import numpy as np
from scipy.signal import filtfilt
from pylops import ricker, PoststackLinearModelling
from matplotlib import pyplot as plt
from mpi4py import MPI
from pylops_mpi import MPIBlockDiag, DistributedArray

plt.close("all")
np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

###############################################################################
# Let's start by defining the important parameters required to model the
# ``pylops.avo.poststack.PoststackLinearModelling`` operator.

# Model
nt0 = 301
dt0 = 0.004
t0 = np.arange(nt0) * dt0
model = np.load("../testdata/poststack_model.npz")
x = model['x'][::3] / 1000.0
z = model['z'] / 1000.0
nx, nz, ny_i = len(x), len(z), 10
m = np.log(model['model'][:, ::3])  # shape=(nz, nx)
# Size of y at all ranks
ny = MPI.COMM_WORLD.allreduce(ny_i)
# Extending over first axis
m3d = np.tile(m[np.newaxis, :, :], (ny_i, 1, 1))

# Smooth model
nsmoothz, nsmoothx = 30, 20
mback = filtfilt(np.ones(nsmoothz) / float(nsmoothz), 1, m, axis=0)
mback = filtfilt(np.ones(nsmoothx) / float(nsmoothx), 1, mback, axis=1)
# Extending over first axis
mback3d = np.tile(mback[np.newaxis, :, :], (ny_i, 1, 1))

# wavelet
ntwav = 41
wav, twav, wavc = ricker(t0[:ntwav // 2 + 1], 20)

###############################################################################
# We model the data using both the dense and linear operator version of
# ``pylops.avo.poststack.PostStackLinearModelling`` at each rank and pass
# these operators to the ``pylops_mpi.MPIBlockDiag`` operator. Furthermore,
# we use this MPIBlockDiag to perform forward and adjoint operations of
# each operator at different ranks.

# Linear PostStackLinearModelling
PPop = PoststackLinearModelling(wav, nt0=nz, spatdims=(ny_i, nx))
BDiag = MPIBlockDiag(ops=[PPop, ])
# DistributedArray
m3d_dist = DistributedArray(global_shape=nx * ny * nz)
m3d_dist[:] = m3d.flatten()
# Forward
d_dist = BDiag * m3d_dist
# Adjoint
d_adjoint_dist = BDiag.H * m3d_dist

# Dense PostStackLinearModelling
PPop_dense = PoststackLinearModelling(wav, nt0=nz, spatdims=(ny_i, nx), explicit=True)
BDiag_dense = MPIBlockDiag(ops=[PPop_dense, ])
# Forward
d_dense_dist = BDiag_dense * m3d_dist
d_dense = d_dense_dist.asarray().reshape((ny, nz, nx))
# Adjoint
d_dense_adj_dist = BDiag_dense.H * m3d_dist
d_dense_adj = d_dense_adj_dist.asarray().reshape((ny, nz, nx))

if rank == 0:
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 9))
    axs[0].imshow(d_dense[3, :, :], cmap="gray", vmin=d_dense.min(), vmax=d_dense.max(),
                  extent=(x[0], x[-1], z[-1], z[0]))
    axs[0].set_title("Data")
    axs[0].axis("tight")

    axs[1].imshow(d_dense_adj[3, :, :], cmap="gray", vmin=d_dense_adj.min(), vmax=d_dense_adj.max(),
                  extent=(x[0], x[-1], z[-1], z[0]))
    axs[1].set_title("Adjoint")
    axs[1].axis("tight")

    axs[2].imshow(m3d[3, :, :], cmap="gist_rainbow", vmin=m.min(), vmax=m.max(), extent=(x[0], x[-1], z[-1], z[0]))
    axs[2].set_title("Model")
    axs[2].axis("tight")

    axs[3].imshow(mback3d[3, :, :], cmap="gist_rainbow", vmin=m.min(), vmax=m.max(), extent=(x[0], x[-1], z[-1], z[0]))
    axs[3].set_title("Smooth Model")
    axs[3].axis("tight")
