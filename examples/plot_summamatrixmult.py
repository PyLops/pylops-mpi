r"""
Distributed SUMMA Matrix Multiplication
=======================================
This example shows how to use the :py:class:`pylops_mpi.basicoperators._MPISummaMatrixMult`
operator to perform matrix-matrix multiplication between a matrix :math:`\mathbf{A}`
distributed in 2D blocks across a square process grid and matrices :math:`\mathbf{X}`
and :math:`\mathbf{Y}` distributed in 2D blocks across the same grid. Similarly,
the adjoint operation can be performed with a matrix :math:`\mathbf{Y}` distributed
in the same fashion as matrix :math:`\mathbf{X}`.

Note that whilst the different blocks of matrix :math:`\mathbf{A}` are directly
stored in the operator on different ranks, the matrices :math:`\mathbf{X}` and
:math:`\mathbf{Y}` are effectively represented by 1-D :py:class:`pylops_mpi.DistributedArray`
objects where the different blocks are flattened and stored on different ranks.
Note that to optimize communications, the ranks are organized in a square grid and
blocks of :math:`\mathbf{A}` and :math:`\mathbf{X}` are systematically broadcast
across different ranks during computation - see below for details.
"""

import math
import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt

import pylops_mpi
from pylops import Conj
from pylops_mpi.basicoperators.MatrixMult import (local_block_spit, MPIMatrixMult, active_grid_comm)

plt.close("all")

###############################################################################
# We set the seed such that all processes can create the input matrices filled
# with the same random number. In practical application, such matrices will be
# filled with data that is appropriate that is appropriate the use-case.
np.random.seed(42)


N, M, K = 6, 6, 6
A_shape, x_shape, y_shape= (N, K), (K, M), (N, M)


base_comm = MPI.COMM_WORLD
comm, rank, row_id, col_id, is_active = active_grid_comm(base_comm, N, M)
print(f"Process {base_comm.Get_rank()} is {'active' if is_active else 'inactive'}")


###############################################################################
# We are now ready to create the input matrices for our distributed matrix
# multiplication example. We need to set up:
# - Matrix :math:`\mathbf{A}` of size :math:`N \times K` (the left operand)
# - Matrix :math:`\mathbf{X}` of size :math:`K \times M` (the right operand)  
# - The result will be :math:`\mathbf{Y} = \mathbf{A} \mathbf{X}` of size :math:`N \times M`
#
# For distributed computation, we arrange processes in a square grid of size
# :math:`P' \times P'` where :math:`P' = \sqrt{P}` and :math:`P` is the total 
# number of MPI processes. Each process will own a block of each matrix 
# according to this 2D grid layout.

p_prime = math.isqrt(comm.Get_size())
print(f"Process grid: {p_prime} x {p_prime} = {comm.Get_size()} processes")

# Create global test matrices with sequential values for easy verification
# Matrix A: Each element :math:`A_{i,j} = i \cdot K + j` (row-major ordering)
# Matrix X: Each element :math:`X_{i,j} = i \cdot M + j`  
A_data = np.arange(int(A_shape[0] * A_shape[1])).reshape(A_shape)
x_data = np.arange(int(x_shape[0] * x_shape[1])).reshape(x_shape)

print(f"Global matrix A shape: {A_shape} (N={A_shape[0]}, K={A_shape[1]})")
print(f"Global matrix X shape: {x_shape} (K={x_shape[0]}, M={x_shape[1]})")
print(f"Expected Global result Y shape: ({A_shape[0]}, {x_shape[1]}) = (N, M)")

################################################################################
# Determine which block of each matrix this process should own
# The 2D block distribution ensures:
# - Process at grid position :math:`(i,j)` gets block :math:`\mathbf{A}[i_{start}:i_{end}, j_{start}:j_{end}]`
# - Block sizes are approximately :math:`\lceil N/P' \rceil \times \lceil K/P' \rceil`  with edge processes handling remainder
#
# .. raw:: html
#
#   <div style="text-align: left; font-family: monospace; white-space: pre;">
#   <b>Example: 2x2 Process Grid with 6x6 Matrices</b>
#   
#   Matrix A (6x6):                    Matrix X (6x6):
#   ┌───────────┬───────────┐      ┌───────────┬───────────┐
#   │  0  1  2  │  3  4  5  │      │  0  1  2  │  3  4  5  │
#   │  6  7  8  │  9 10 11  │      │  6  7  8  │  9 10 11  │
#   │ 12 13 14  │ 15 16 17  │      │ 12 13 14  │ 15 16 17  │
#   ├───────────┼───────────┤      ├───────────┼───────────┤
#   │ 18 19 20  │ 21 22 23  │      │ 18 19 20  │ 21 22 23  │
#   │ 24 25 26  │ 27 28 29  │      │ 24 25 26  │ 27 28 29  │
#   │ 30 31 32  │ 33 34 35  │      │ 30 31 32  │ 33 34 35  │
#   └───────────┴───────────┘      └───────────┴───────────┘
#   
#   Process (0,0): A[0:3, 0:3], X[0:3, 0:3]
#   Process (0,1): A[0:3, 3:6], X[0:3, 3:6]  
#   Process (1,0): A[3:6, 0:3], X[3:6, 0:3]
#   Process (1,1): A[3:6, 3:6], X[3:6, 3:6]
#   </div>
#

A_slice = local_block_spit(A_shape, rank, comm)
x_slice = local_block_spit(x_shape, rank, comm)
################################################################################
# Extract the local portion of each matrix for this process
A_local = A_data[A_slice]
x_local = x_data[x_slice]

print(f"Process {rank}: A_local shape {A_local.shape}, X_local shape {x_local.shape}")
print(f"Process {rank}: A slice {A_slice}, X slice {x_slice}")

x_dist = pylops_mpi.DistributedArray(global_shape=(K * M),
                                     local_shapes=comm.allgather(x_local.shape[0] * x_local.shape[1]),
                                     base_comm=comm,
                                     partition=pylops_mpi.Partition.SCATTER,
                                     dtype=x_local.dtype)
x_dist[:] = x_local.flatten()

################################################################################
# We are now ready to create the SUMMA :py:class:`pylops_mpi.basicoperators.MPIMatrixMult`
# operator and the input matrix :math:`\mathbf{X}`. Given that we chose a block-block distribution
# of data we shall use SUMMA
Aop = MPIMatrixMult(A_local, M, base_comm=comm, kind="summa", dtype=A_local.dtype)

################################################################################
# We can now apply the forward pass :math:`\mathbf{y} = \mathbf{Ax}` (which
# effectively implements a distributed matrix-matrix multiplication
# :math:`Y = \mathbf{AX}`). Note :math:`\mathbf{Y}` is distributed in the same
# way as the input :math:`\mathbf{X}` in a block-block fashion.
y_dist = Aop @ x_dist

###############################################################################
# Next we apply the adjoint pass :math:`\mathbf{x}_{adj} = \mathbf{A}^H \mathbf{x}`
# (which effectively implements a distributed summa matrix-matrix multiplication
# :math:`\mathbf{X}_{adj} = \mathbf{A}^H \mathbf{X}`). Note that
# :math:`\mathbf{X}_{adj}` is again distributed in the same way as the input
# :math:`\mathbf{X}` in a block-block fashion.
xadj_dist = Aop.H @ y_dist

###############################################################################
# Finally, we show that the SUMMA :py:class:`pylops_mpi.basicoperators.MPIMatrixMult`
# operator can be combined with any other PyLops-MPI operator. We are going to
# apply here a conjugate operator to the output of the matrix multiplication.
Dop = Conj(dims=(A_local.shape[0], x_local.shape[1]))
DBop = pylops_mpi.MPIBlockDiag(ops=[Dop, ])
Op = DBop @ Aop
y1 = Op @ x_dist
