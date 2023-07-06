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
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

###############################################################################
# Let's start by defining the important parameters required to model the
# ``pylops.avo.poststack.PoststackLinearModelling`` operator.

# Model
model = np.load("../testdata/avo/poststack_model.npz")
x, z, m = model['x'], model['z'], np.log(model['model'])

# Making m a 3-D
ny_i = 10
y = np.arange(ny_i)
m3d_i = np.tile(m[:, :, np.newaxis], (1, 1, ny_i)).transpose((2, 1, 0))
ny_i, nx, nz = m3d_i.shape

# Size of y at all ranks
ny = MPI.COMM_WORLD.allreduce(ny_i)

# Smooth model
nsmoothz, nsmoothx = 30, 20
mback = filtfilt(np.ones(nsmoothz) / float(nsmoothz), 1, m, axis=0)
mback = filtfilt(np.ones(nsmoothx) / float(nsmoothx), 1, mback, axis=1)
# Making mback a 3-D
mback3d_i = np.tile(mback[:, :, np.newaxis], (1, 1, ny_i)).transpose((2, 1, 0))

# wavelet
dt = 0.004
t0 = np.arange(nz) * dt
ntwav = 41
wav = ricker(t0[:ntwav // 2 + 1], 15)[0]

# Collecting all the m3d and mback3d at all ranks
m3d = np.concatenate(MPI.COMM_WORLD.allgather(m3d_i))
mback3d = np.concatenate(MPI.COMM_WORLD.allgather(mback3d_i))

###############################################################################
# We model the data using both the dense and linear operator version of
# ``pylops.avo.poststack.PostStackLinearModelling`` at each rank and pass
# these operators to the ``pylops_mpi.MPIBlockDiag`` operator. Furthermore,
# we use this MPIBlockDiag to perform forward operation on each operator
# at different ranks to get the ``data``.

# Flattening modelling data
m3d_dist = DistributedArray(global_shape=nx * ny * nz)
m3d_dist[:] = m3d_i.transpose((2, 0, 1)).flatten()

# Linear PostStackLinearModelling
PPop = PoststackLinearModelling(wav, nt0=nz, spatdims=(ny_i, nx))
BDiag = MPIBlockDiag(ops=[PPop, ])

# Data
d_dist = BDiag * m3d_dist
d_local = d_dist.local_array.reshape((nz, ny_i, nx)).transpose(1, 2, 0)
d = d_dist.asarray().reshape((nz, ny, nx)).transpose(1, 2, 0)

# Dense PostStackLinearModelling
PPop_dense = PoststackLinearModelling(wav, nt0=nz, spatdims=(ny_i, nx), explicit=True)
BDiag_dense = MPIBlockDiag(ops=[PPop_dense, ])

# Dense Data
d_dense_dist = BDiag_dense * m3d_dist
d_dense_local = d_dense_dist.local_array.reshape((nz, ny_i, nx)).transpose(1, 2, 0)
d_dense = d_dense_dist.asarray().reshape((nz, ny, nx)).transpose(1, 2, 0)

if rank == 0:
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9, 12), constrained_layout=True)
    axs[0][0].imshow(d_dense[5, :, :].T, cmap="gray", vmin=-1, vmax=1)
    axs[0][0].set_title("Data x-z")
    axs[0][0].axis("tight")
    axs[0][1].imshow(d_dense[:, 400, :].T, cmap='gray', vmin=-1, vmax=1)
    axs[0][1].set_title('Data y-z')
    axs[0][1].axis('tight')
    axs[0][2].imshow(d_dense[:, :, 220].T, cmap='gray', vmin=-1, vmax=1)
    axs[0][2].set_title('Data x-y')
    axs[0][2].axis('tight')

    axs[1][0].imshow(m3d[5, :, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[1][0].set_title("Model x-z")
    axs[1][0].axis("tight")
    axs[1][1].imshow(m3d[:, 400, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[1][1].set_title("Model y-z")
    axs[1][1].axis("tight")
    axs[1][2].imshow(m3d[:, :, 220].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[1][2].set_title("Model y-z")
    axs[1][2].axis("tight")

    axs[2][0].imshow(mback3d[5, :, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[2][0].set_title("Smooth Model x-z")
    axs[2][0].axis("tight")
    axs[2][1].imshow(mback3d[:, 400, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[2][1].set_title("Smooth Model y-z")
    axs[2][1].axis("tight")
    axs[2][2].imshow(mback3d[:, :, 220].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[2][2].set_title("Smooth Model y-z")
    axs[2][2].axis("tight")
