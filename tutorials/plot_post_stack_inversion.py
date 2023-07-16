r"""
Post Stack Inversion - 3D
=========================
This tutorial demonstrates the implementation of a distributed 3D Post-stack inversion. It consists
of a first part showing how to model a 3D synthetic post-stack seismic data from a 3D model of the 
subsurface acoustic impedence in a distributed manner, following by a second part when inversion
is carried out. 

This tutorial builds on the :py:class:`pylops.avo.poststack.PoststackLinearModelling`
operator to model 1d post-stack seismic traces 1d profiles of the subsurface acoustic impedence
by means of the following equation

.. math::
    d(t) = \frac{1}{2} w(t) * \frac{\mathrm{d}\ln \text{AI}(t)}{\mathrm{d}t}

where :math:`\text{AI}(t)` is the acoustic impedance profile and :math:`w(t)` is
the time domain seismic wavelet. Being this inherently a 1d operator, we can easily
set up a problem where one of the dimensions (here the y-dimension) is distributed 
across ranks and each of them is in charge of performing modelling for a subvolume of
the entire domain. Using a compact matrix-vector notation, the entire problem can
be written as

.. math::
    \begin{bmatrix}
        \mathbf{d}_{1}  \\
        \mathbf{d}_{2}  \\
        \vdots     \\
        \mathbf{d}_{N}
    \end{bmatrix} =
    \begin{bmatrix}
        \mathbf{G}_1  & \mathbf{0}   &  \ldots &  \mathbf{0}  \\
        \mathbf{0}    & \mathbf{G}_2 &  \ldots &  \mathbf{0}  \\
        \vdots        & \vdots       &  \ddots &  \vdots         \\
        \mathbf{0}    & \mathbf{0}   &  \ldots &  \mathbf{G}_N
        \end{bmatrix}
    \begin{bmatrix}
        \mathbf{ai}_{1}  \\
        \mathbf{ai}_{2}  \\
        \vdots     \\
        \mathbf{ai}_{N}
    \end{bmatrix} 

where :math:`\mathbf{G}_i` is a post-stack modelling operator, :math:`\mathbf{d}_i` 
is the data, and :math:`\mathbf{ai}_i` is the input model for the i-th portion of the model.

This problem can be easily set up using the :py:class:`pylops_mpi.basicoperators.MPIBlockDiag`
operator.

"""

import numpy as np
from scipy.signal import filtfilt
from matplotlib import pyplot as plt
from mpi4py import MPI

from pylops.utils.wavelets import ricker
from pylops.basicoperators import SecondDerivative, Transpose, VStack
from pylops.avo.poststack import PoststackLinearModelling
from pylops_mpi import MPIBlockDiag, DistributedArray, cgls

plt.close("all")
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

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
# :py:func:`pylops.avo.poststack.PoststackLinearModelling` at each rank to model a
# subset of the data along the y-axis. Such operators are passed
# to the :py:class:`pylops_mpi.basicoperators.MPIBlockDiag` operator, which is then used to perform
# the different forward operations of each individual operator
# at different ranks to compute the overall data. Note that to simplify the
# handling of the model and data, we split and distribute the first axis,
# and use :py:class:`pylops.Transpose` to rearrange the model and data
# in the form required by the :py:func:`pylops.avo.poststack.PoststackLinearModelling`
# operator.

# Create flattened model data
m3d_dist = DistributedArray(global_shape=ny * nx * nz)
m3d_dist[:] = m3d_i.flatten()

# Create flattened smooth model data
mback3d_dist = DistributedArray(global_shape=ny * nx * nz)
mback3d_dist[:] = mback3d_i.flatten()

# LinearOperator PostStackLinearModelling
PPop = PoststackLinearModelling(wav, nt0=nz, spatdims=(ny_i, nx))
Top = Transpose((ny_i, nx, nz), (2, 0, 1))
BDiag = MPIBlockDiag(ops=[Top.H @ PPop @ Top, ])

# Data
d_dist = BDiag @ m3d_dist
d_local = d_dist.local_array.reshape((ny_i, nx, nz))
d = d_dist.asarray().reshape((ny, nx, nz))
d_0_dist = BDiag @ mback3d_dist
d_0 = d_dist.asarray().reshape((ny, nx, nz))

###############################################################################
# We perform 2 different kinds of inversions:
#
# * Inversion calculated iteratively using the :py:class:`pylops_mpi.optimization.cls_basic.CGLS` solver.
#
# * Inversion with spatial regularization along the non-distributed dimensions (x and z).
#   This requires extending the operator and data of each rank in the following manner
#
#  .. math::
#    \begin{bmatrix}
#        \mathbf{d}_{i}  \\
#        \mathbf{0}
#    \end{bmatrix} =
#    \begin{bmatrix}
#        \mathbf{G}_i \\
#        \epsilon \mathbf{D}_{2,x} \\
#        \epsilon \mathbf{D}_{2,z} \\
#    \end{bmatrix} \mathbf{ai}_{i}
#
# where :math:`\mathbf{D}_{2,x}` and :math:`\mathbf{D}_{2,z}` apply the second derivative over the z- and x-axes.

# Inversion using CGLS solver
minv3d_iter_dist = cgls(BDiag, d_dist, x0=mback3d_dist, niter=100, show=True)[0]
minv3d_iter = minv3d_iter_dist.asarray().reshape((ny, nx, nz))

# Regularized inversion
epsR = 1e1
Dz = SecondDerivative((ny_i, nx, nz), axis=-1)
Dx = SecondDerivative((ny_i, nx, nz), axis=-2)

d_dist_reg = DistributedArray(global_shape=3 * ny * nx * nz)
d_dist_reg[:ny_i*nz*nx] = d_dist.local_array
d_dist_reg[ny_i*nz*nx:] = 0.
BDiag_reg = MPIBlockDiag(ops=[VStack([Top.H @ PPop @ Top, epsR * Dx, epsR * Dz]),])
minv3d_reg_dist = cgls(BDiag_reg, d_dist_reg, x0=mback3d_dist, niter=100, show=True)[0]
minv3d_reg = minv3d_reg_dist.asarray().reshape((ny, nx, nz))

###############################################################################
# Finally, we display the modeling and inversion results

if rank == 0:
    # Check the distributed implementation gives the same result
    # as the one running only on rank0
    PPop0 = PoststackLinearModelling(wav, nt0=nz, spatdims=(ny, nx))
    d0 = (PPop0 @ m3d.transpose(2, 0, 1)).transpose(1, 2, 0)
    d0_0 = (PPop0 @ m3d.transpose(2, 0, 1)).transpose(1, 2, 0)

    # Check the two distributed implementations give the same modelling results
    print('Distr == Local', np.allclose(d, d0))
    print('Smooth Distr == Local', np.allclose(d_0, d0_0))

    # Visualize
    fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(9, 14), constrained_layout=True)
    axs[0][0].imshow(m3d[5, :, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[0][0].set_title("Model x-z")
    axs[0][0].axis("tight")
    axs[0][1].imshow(m3d[:, 200, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[0][1].set_title("Model y-z")
    axs[0][1].axis("tight")
    axs[0][2].imshow(m3d[:, :, 220].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[0][2].set_title("Model y-z")
    axs[0][2].axis("tight")

    axs[1][0].imshow(mback3d[5, :, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[1][0].set_title("Smooth Model x-z")
    axs[1][0].axis("tight")
    axs[1][1].imshow(mback3d[:, 200, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[1][1].set_title("Smooth Model y-z")
    axs[1][1].axis("tight")
    axs[1][2].imshow(mback3d[:, :, 220].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[1][2].set_title("Smooth Model y-z")
    axs[1][2].axis("tight")

    axs[2][0].imshow(d[5, :, :].T, cmap="gray", vmin=-1, vmax=1)
    axs[2][0].set_title("Data x-z")
    axs[2][0].axis("tight")
    axs[2][1].imshow(d[:, 200, :].T, cmap='gray', vmin=-1, vmax=1)
    axs[2][1].set_title('Data y-z')
    axs[2][1].axis('tight')
    axs[2][2].imshow(d[:, :, 220].T, cmap='gray', vmin=-1, vmax=1)
    axs[2][2].set_title('Data x-y')
    axs[2][2].axis('tight')

    axs[3][0].imshow(minv3d_iter[5, :, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[3][0].set_title("Inverted Model iter x-z")
    axs[3][0].axis("tight")
    axs[3][1].imshow(minv3d_iter[:, 200, :].T, cmap='gist_rainbow', vmin=m.min(), vmax=m.max())
    axs[3][1].set_title('Inverted Model iter y-z')
    axs[3][1].axis('tight')
    axs[3][2].imshow(minv3d_iter[:, :, 220].T, cmap='gist_rainbow', vmin=m.min(), vmax=m.max())
    axs[3][2].set_title('Inverted Model iter x-y')
    axs[3][2].axis('tight')

    axs[4][0].imshow(minv3d_reg[5, :, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
    axs[4][0].set_title("Regularized Inverted Model iter x-z")
    axs[4][0].axis("tight")
    axs[4][1].imshow(minv3d_reg[:, 200, :].T, cmap='gist_rainbow', vmin=m.min(), vmax=m.max())
    axs[4][1].set_title('Regularized Inverted Model iter y-z')
    axs[4][1].axis('tight')
    axs[4][2].imshow(minv3d_reg[:, :, 220].T, cmap='gist_rainbow', vmin=m.min(), vmax=m.max())
    axs[4][2].set_title('Regularized Inverted Model iter x-y')
    axs[4][2].axis('tight')
