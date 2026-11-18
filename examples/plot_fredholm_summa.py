"""
Tiled Fredholm integral with SUMMA
=================================

Apply a different kernel to each model slice with
:class:`pylops_mpi.signalprocessing.MPIFredholm1SUMMA`. Run this example
with four ranks for one 2-by-2 grid, or eight for two independent grids.
"""

import numpy as np
from mpi4py import MPI

from pylops_mpi import DistributedArray, Partition
from pylops_mpi.basicoperators.MatrixMult import local_block_split
from pylops_mpi.signalprocessing import MPIFredholm1SUMMA

###############################################################################
# Each group owns two slices. Use a 2-by-2 grid when the communicator size
# permits it and one rank per group otherwise.

comm = MPI.COMM_WORLD
grid_size = 4 if comm.size % 4 == 0 else 1
pb = comm.size // grid_size
batch, grid_rank = divmod(comm.rank, grid_size)
grid = comm.Split(color=batch, key=grid_rank)

nx, ny, nz = 5, 4, 3
rng = np.random.default_rng(batch)
kernel = rng.normal(size=(2, nx, ny))
model = rng.normal(size=(2, ny, nz))

###############################################################################
# The kernel and model have different row and column axes. Select each
# rank's two-dimensional block, keeping both of its slices together.

kernel_block = local_block_split((nx, ny), grid_rank, grid)
model_block = local_block_split((ny, nz), grid_rank, grid)
data_block = local_block_split((nx, nz), grid_rank, grid)
grid.Free()

local_kernel = kernel[(slice(None), *kernel_block)]
local_model = model[(slice(None), *model_block)]
op = MPIFredholm1SUMMA(local_kernel, nz=nz, pb=pb)
x = DistributedArray(
    global_shape=2 * pb * ny * nz,
    local_shapes=comm.allgather(local_model.size),
    partition=Partition.SCATTER,
)
x[:] = local_model.ravel()

###############################################################################
# The output uses the same batch and grid, with rows over x and columns over z.
# Check the local values against the corresponding block of the dense product.

y = op @ x
expected = (kernel @ model)[(slice(None), *data_block)]
np.testing.assert_allclose(y.local_array.reshape(expected.shape), expected)
comm.Barrier()
if comm.rank == 0:
    print(f"Verified all output tiles across {pb} batch group(s).")
