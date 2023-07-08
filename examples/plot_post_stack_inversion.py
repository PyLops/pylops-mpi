"""
Post Stack Inversion - 3D
=========================
This illustration demonstrates the implementation of a distributed 3D Post-stack inversion. It involves
modelling a 3D synthetic post-stack seismic data from a 3D model of the subsurface acoustic impedence.
"""

import numpy as np
from scipy.signal import filtfilt
from matplotlib import pyplot as plt
from mpi4py import MPI

from pylops.utils.wavelets import ricker
from pylops.basicoperators import Transpose
from pylops.avo.poststack import PoststackLinearModelling
from pylops_mpi import MPIBlockDiag, DistributedArray

plt.close("all")
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

###############################################################################
# Let's start by defining all of the parameters required by the
# ``pylops.avo.poststack.PoststackLinearModelling`` operator.

# Model
model = np.load("../testdata/avo/poststack_model.npz")
x, z, m = model['x'], model['z'], np.log(model['model'])

# Making m a 3D model
ny_i = 20  # size of model in y direction for rank i
y = np.arange(ny_i)
m3d_i = np.tile(m[:, :, np.newaxis], (1, 1, ny_i)).transpose((2, 1, 0))
ny_i, nx, nz = m3d_i.shape

# Size of y at all ranks
ny = MPI.COMM_WORLD.allreduce(ny_i)

# Smooth model
nsmoothy, nsmoothx, nsmoothz = 5, 30, 20
mback3d_i = filtfilt(np.ones(nsmoothy) / float(nsmoothy), 1, m3d_i, axis=0)
mback3d_i = filtfilt(np.ones(nsmoothx) / float(nsmoothx), 1, mback3d_i, axis=1)
mback3d_i = filtfilt(np.ones(nsmoothz) / float(nsmoothz), 1, mback3d_i, axis=2)

# Wavelet
dt = 0.004
t0 = np.arange(nz) * dt
ntwav = 41
wav = ricker(t0[:ntwav // 2 + 1], 15)[0]

# Collecting all the m3d and mback3d at all ranks
m3d = np.concatenate(MPI.COMM_WORLD.allgather(m3d_i))
mback3d = np.concatenate(MPI.COMM_WORLD.allgather(mback3d_i))

###############################################################################
# We now create the linear operator version of
# ``pylops.avo.poststack.PostStackLinearModelling`` at each rank to model a
# subset of the data along the y-axis. Such operators are passes
# to the ``pylops_mpi.MPIBlockDiag`` operator, which is then used to perform
# the different forward operations of each individual operator
# at different ranks to compute the overall data. Note that to simplify the
# handling of the model and data, we split and distribute the first axis,
# and use ``pylops.basicoperators.Transpose`` to rearrange the model and data
# in the form required by the ``pylops.avo.poststack.PostStackLinearModelling``
# operator

# Create flattened model data
m3d_dist = DistributedArray(global_shape=ny * nx * nz)
m3d_dist[:] = m3d_i.flatten()

# LinearOperator PostStackLinearModelling
PPop = PoststackLinearModelling(wav, nt0=nz, spatdims=(ny_i, nx))
Top = Transpose((ny_i, nx, nz), (2, 0, 1))
BDiag = MPIBlockDiag(ops=[Top.H @ PPop @ Top, ])

# Data
d_dist = BDiag @ m3d_dist
d_local = d_dist.local_array.reshape((ny_i, nx, nz))
d = d_dist.asarray().reshape((ny, nx, nz))

if rank == 0:
    # Check the distributed implementation gives the same result
    # as the one running only on rank0
    PPop0 = PoststackLinearModelling(wav, nt0=nz, spatdims=(ny, nx))
    d0 = (PPop0 @ m3d.transpose(2, 0, 1)).transpose(1, 2, 0)

    # Check the two distributed implementations give the same result
    print('Distr == Local', np.allclose(d, d0))

    # Visualize
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9, 12), constrained_layout=True)
    axs[0][0].imshow(m3d[5, :, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[0][0].set_title("Model x-z")
    axs[0][0].axis("tight")
    axs[0][1].imshow(m3d[:, 400, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[0][1].set_title("Model y-z")
    axs[0][1].axis("tight")
    axs[0][2].imshow(m3d[:, :, 220].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[0][2].set_title("Model y-z")
    axs[0][2].axis("tight")

    axs[1][0].imshow(mback3d[5, :, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[1][0].set_title("Smooth Model x-z")
    axs[1][0].axis("tight")
    axs[1][1].imshow(mback3d[:, 400, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[1][1].set_title("Smooth Model y-z")
    axs[1][1].axis("tight")
    axs[1][2].imshow(mback3d[:, :, 220].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[1][2].set_title("Smooth Model y-z")
    axs[1][2].axis("tight")

    axs[2][0].imshow(d[5, :, :].T, cmap="gray", vmin=-1, vmax=1)
    axs[2][0].set_title("Data x-z")
    axs[2][0].axis("tight")
    axs[2][1].imshow(d[:, 400, :].T, cmap='gray', vmin=-1, vmax=1)
    axs[2][1].set_title('Data y-z')
    axs[2][1].axis('tight')
    axs[2][2].imshow(d[:, :, 220].T, cmap='gray', vmin=-1, vmax=1)
    axs[2][2].set_title('Data x-y')
    axs[2][2].axis('tight')

    plt.savefig('./plots/Poststack.png')
