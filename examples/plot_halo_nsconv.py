from matplotlib import pyplot as plt

import sys
import math
import time
import numpy as np
import pylops

from pylops.utils.wavelets import ricker
from pylops_mpi.basicoperators.Halo import MPIHalo, halo_block_split
from mpi4py import MPI

import pylops_mpi

plt.close("all")
np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def pause(comm, t=4):
    sys.stdout.flush()
    comm.barrier()
    time.sleep(t)

def local_extent_from_slice(local_shape, local_slice, halo):
    lefts = []
    rights = []
    for sl in local_slice:
        lefts.append(halo if (sl.start or 0) > 0 else 0)
        rights.append(halo if sl.stop is not None else 0)
    extent = tuple(dim + l + r for dim, l, r in zip(local_shape, lefts, rights))
    return extent, lefts, rights


# Input signal operator
nlocal = 64
nfilters_local = 2
n = nlocal * size
proc_grid_shape = (size, )

# Filters
ntw = 16
dt = 0.004
tw = np.arange(ntw) * dt
fs = np.arange(nfilters_local * size) * 8 + 20
wavs = np.stack([ricker(tw, f0=f)[0] for f in fs])

n_between_h = nlocal // nfilters_local
ih = nlocal // (2 * nfilters_local) + np.arange(0, nlocal * size, n_between_h)
print(ih, n_between_h)

t = np.arange(n) * dt
x = np.zeros(n, dtype=np.float64)
x[ih] = 1.0 # no mismatch in forward, some in adjoint...
# x = np.random.normal(0, 1, n)  # more mismatch

if rank == 0:
    plt.figure()
    plt.plot(x)
    plt.savefig('x.png')

    plt.figure()
    plt.plot(wavs.T)
    plt.savefig('wavs.png')

Cop = pylops.signalprocessing.NonStationaryConvolve1D(
    dims=n, hs=wavs, ih=ih
)

y = Cop @ x
xadj = Cop.H @ y

if rank == 0:
    plt.figure(figsize=(10, 3))
    plt.plot(t, x, "k")
    plt.plot(t, y, "k")
    plt.xlabel("Time [sec]")
    plt.title("Input and output")
    plt.xlim(0, t[-1])
    plt.savefig('y.png')

halo = n_between_h
halo_op = MPIHalo(
    dims=(n, ),
    halo=halo,
    proc_grid_shape=proc_grid_shape,
    comm=comm,
    normalize=True,
)

# Distributed array
x_dist = pylops_mpi.DistributedArray(
    global_shape=n,
    base_comm=comm,
    partition=pylops_mpi.Partition.SCATTER)
x_slice = halo_block_split((n, ), comm, proc_grid_shape)
x_local = x[x_slice]
x_dist.local_array[:] = x_local

# Apply halo
x_dist_halo = halo_op @ x_dist

if rank == 0:
    print('1D Halo with halo=1')
pause(comm, t=1)
print(f"Rank {rank} - shape before/after haloing: {x_dist.local_array.size}"
      f"/{x_dist_halo.local_array.size}")

x_slice = halo_block_split((n, ), comm, proc_grid_shape)
x_local = x[x_slice]
x_local_shape = x_local.shape

xhalo_local_shape, lefts, rights = local_extent_from_slice(x_local_shape, x_slice, halo)

print(f"Rank {rank}, xhalo_local_shape={xhalo_local_shape}")

if rank == 0:
    COp = pylops.signalprocessing.NonStationaryConvolve1D(
        dims=nlocal + halo, hs=wavs[:nfilters_local + 1], ih=ih[:nfilters_local + 1]
    )
elif rank == size - 1:
    COp = pylops.signalprocessing.NonStationaryConvolve1D(
        dims=nlocal + halo, hs=wavs[-nfilters_local - 1:], ih=ih[-nfilters_local - 1:] - x_slice[0].start + halo
    )
else:
    COp = pylops.signalprocessing.NonStationaryConvolve1D(
        dims=nlocal + 2 * halo, hs=wavs[nfilters_local * rank - 1:nfilters_local * (rank + 1) + 1],
        ih=ih[nfilters_local * rank - 1:nfilters_local * (rank + 1) + 1] - x_slice[0].start + halo
    )

COp_dist = pylops_mpi.MPIBlockDiag([COp, ])
Op_dist = halo_op.H @ COp_dist @ halo_op

y_dist = Op_dist @ x_dist
xadj_dist = Op_dist.H @ y_dist

y1 = y_dist.asarray()
xadj1 = xadj_dist.asarray()


if rank == 0:
    plt.figure(figsize=(10, 3))
    plt.plot(t, x, "k")   
    plt.plot(t, y1, "k")
    plt.plot(t, y, "--r")
    plt.plot(t, y - y1, "g")
    plt.xlabel("Time [sec]")
    plt.title("Input and output")
    plt.xlim(0, t[-1])
    plt.savefig('y1.png')

    plt.figure(figsize=(10, 3))
    plt.plot(t, y - y1, "g")
    plt.xlabel("Time [sec]")
    plt.title("Adjoint")
    plt.xlim(0, t[-1])
    plt.savefig('yerr.png')

if rank == 0:
    print('xadj', xadj.shape)
    print('xadj1', xadj1.shape)
    print('xadj - xadj1', xadj - xadj1)

    plt.figure(figsize=(10, 3))
    plt.plot(t, x, "k")   
    plt.plot(t, xadj, "k")   
    plt.plot(t, xadj1, "--r")
    plt.plot(t, xadj - xadj1, "g")
    plt.xlabel("Time [sec]")
    plt.title("Adjoint")
    plt.xlim(0, t[-1])
    plt.savefig('xadj.png')

    plt.figure(figsize=(10, 3))
    plt.plot(t, xadj - xadj1, "g")
    plt.xlabel("Time [sec]")
    plt.title("Adjoint")
    plt.xlim(0, t[-1])
    plt.savefig('xadjerr.png')

assert np.allclose(y, y1)
assert np.allclose(xadj, xadj1, atol=1e-1, rtol=1e-3)


# from pylops_mpi.utils.dottest import dottest

# x_dist = pylops_mpi.DistributedArray(
#     global_shape=n,
#     base_comm=comm,
#     partition=pylops_mpi.Partition.SCATTER,
#     dtype=np.float64,
# )
# x_dist[:] = np.random.normal(0.0, 1.0, x_dist.local_array.shape)

# y_dist = pylops_mpi.DistributedArray(
#     global_shape=n,
#     base_comm=comm,
#     partition=pylops_mpi.Partition.SCATTER,
#     dtype=np.float64,
# )
# y_dist[:] = np.random.normal(0.0, 1.0, y_dist.local_array.shape)

# dottest(Op_dist, x_dist, y_dist)
