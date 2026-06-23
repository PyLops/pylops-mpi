r"""
Reflectivity Inversion - 3D with CUDA-Aware MPI
=============================================
This tutorial is an extension of the :ref:`sphx_glr_tutorials_reflectivity.py` 
tutorial where PyLops-MPI is run in multi-GPU setting with GPUs communicating 
via MPI.
"""


import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
from mpi4py import MPI

from pylops.utils.wavelets import ricker
from pylops.basicoperators import FirstDerivative
from pylops.signalprocessing import Convolve1D

import pylops_mpi

plt.close("all")
rank = MPI.COMM_WORLD.Get_rank()
device_count = cp.cuda.runtime.getDeviceCount()
cp.cuda.Device(rank % device_count).use();

###############################################################################
# Let's start by defining all the parameters required by the
# :py:func:`pylops.avo.poststack.PoststackLinearModelling` operator.

# Model
model = np.load("../testdata/avo/poststack_model.npz")
x, z, m = model['x'][::3], model['z'], np.log(model['model'])[:, ::3]

# Making m a 3D model
ny_i = 20  # size of model in y direction for rank i
y = np.arange(ny_i)
m3d_i = np.tile(m[:, :, np.newaxis], (1, 1, ny_i)).transpose((2, 1, 0))
ny_i, nx, nz = m3d_i.shape

# Size of y at all ranks
ny = MPI.COMM_WORLD.allreduce(ny_i)

# Wavelet
dt = 0.004
t0 = np.arange(nz) * dt
ntwav = 41
wav, _, wavc = ricker(t0[:ntwav // 2 + 1], 15)

# Collecting m3d at all ranks
m3d = np.concatenate(MPI.COMM_WORLD.allgather(m3d_i))

###############################################################################
# We now create the linear operators to model the data (including a time derivative
# as in the post-stack tutorial) as well as that to invert the data for the
# underlying reflectivity model.

# Create flattened model data
m3d_dist = pylops_mpi.DistributedArray(global_shape=ny * nx * nz, engine="cupy")
m3d_dist[:] = cp.asarray(m3d_i.flatten())

# LinearOperator Derivative + Convolve
Dop = FirstDerivative((ny_i, nx, nz), axis=-1)
Cop = Convolve1D((ny_i, nx, nz), wav, offset=wavc, axis=-1)
DDiag = pylops_mpi.basicoperators.MPIBlockDiag(ops=[Dop, ])
CDiag = pylops_mpi.basicoperators.MPIBlockDiag(ops=[Cop, ])

# Reflectivity
r_dist = DDiag @ m3d_dist
r_local = r_dist.local_array.reshape((ny_i, nx, nz))
r = r_dist.asarray().reshape((ny, nx, nz))

# Data
d_dist = CDiag @ r_dist
d_local = d_dist.local_array.reshape((ny_i, nx, nz))
d = d_dist.asarray().reshape((ny, nx, nz))

###############################################################################
# We now perform sparsity-promotion inversion

r0_dist = pylops_mpi.DistributedArray(global_shape=ny * nx * nz, engine="cupy")
r0_dist[:] = 0.

rinv3d_dist = pylops_mpi.optimization.sparsity.ista(
    CDiag, d_dist, x0=r0_dist,
    niter=400, eps=1e-2, tol=1e-8, show=False)[0]
rinv3d = rinv3d_dist.asarray().reshape((ny, nx, nz))

###############################################################################
# Finally, we display the modeling and inversion results

if rank == 0:
    # Check the distributed implementation gives the same result
    # as the one running only on rank0
    Dop0 = FirstDerivative((ny, nx, nz), axis=-1)
    Cop0 = Convolve1D((ny, nx, nz), wav, offset=wavc, axis=-1)

    r0 = Dop0 @ m3d
    d0 = Cop0 @ r0

    # Check the two distributed implementations give the same modelling results
    print('Reflectivity Distr == Local', np.allclose(cp.asnumpy(d), d0, atol=1e-6))
    print('Data Distr == Local', np.allclose(cp.asnumpy(r), r0, atol=1e-6))

    # Visualize
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(9, 14), constrained_layout=True)
    axs[0][0].imshow(m3d[5, :, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[0][0].set_title("Model x-z")
    axs[0][0].axis("tight")
    axs[0][1].imshow(m3d[:, 200, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[0][1].set_title("Model y-z")
    axs[0][1].axis("tight")
    axs[0][2].imshow(m3d[:, :, 220].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[0][2].set_title("Model y-z")
    axs[0][2].axis("tight")
    
    axs[1][0].imshow(r[5, :, :].T.get(), cmap="gray", vmin=-.1, vmax=.1)
    axs[1][0].set_title("Reflectivity Model x-z")
    axs[1][0].axis("tight")
    axs[1][1].imshow(r[:, 200, :].T.get(), cmap="gray", vmin=-.1, vmax=.1)
    axs[1][1].set_title("Reflectivity Model y-z")
    axs[1][1].axis("tight")
    axs[1][2].imshow(r[:, :, 220].T.get(), cmap="gray", vmin=-.1, vmax=.1)
    axs[1][2].set_title("Reflectivity Model y-z")
    axs[1][2].axis("tight")
    
    axs[2][0].imshow(d[5, :, :].T.get(), cmap="gray", vmin=-1, vmax=1)
    axs[2][0].set_title("Data x-z")
    axs[2][0].axis("tight")
    axs[2][1].imshow(d[:, 200, :].T.get(), cmap='gray', vmin=-1, vmax=1)
    axs[2][1].set_title('Data y-z')
    axs[2][1].axis('tight')
    axs[2][2].imshow(d[:, :, 220].T.get(), cmap='gray', vmin=-1, vmax=1)
    axs[2][2].set_title('Data x-y')
    axs[2][2].axis('tight')
    
    axs[3][0].imshow(rinv3d[5, :, :].T.get(), cmap='gray', vmin=-.1, vmax=.1)
    axs[3][0].set_title("Inverted Reflectivity iter x-z")
    axs[3][0].axis("tight")
    axs[3][1].imshow(rinv3d[:, 200, :].T.get(), cmap='gray', vmin=-.1, vmax=.1)
    axs[3][1].set_title('Inverted Reflectivity iter y-z')
    axs[3][1].axis('tight')
    axs[3][2].imshow(rinv3d[:, :, 220].T.get(), cmap='gray', vmin=-.1, vmax=.1)
    axs[3][2].set_title('Inverted Reflectivity iter x-y')
    axs[3][2].axis('tight')