"""
Distributed Matrix Multiplication
=================================
This example shows how to use the :py:class:`pylops_mpi.basicoperators.MatrixMult.MPIMatrixMult`.
This class provides a way to distribute arrays across multiple processes in
a parallel computing environment.
"""
from matplotlib import pyplot as plt
import math
import numpy as np
from mpi4py import MPI

from pylops_mpi import DistributedArray, Partition
from pylops_mpi.basicoperators.MatrixMult import MPIMatrixMult

plt.close("all")
###############################################################################
# We set the seed such that all processes initially start out with the same initial matrix.
# Ideally this data would be loaded in a manner appropriate to the use-case.
np.random.seed(42)

# MPI parameters
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # rank of current process
size = comm.Get_size() # number of processes

p_prime = int(math.ceil(math.sqrt(size)))
C = int(math.ceil(size / p_prime))

if (p_prime * C) != size:
    print("No. of procs has to be a square number")
    exit(-1)

# matrix dims
M, K, N = 4, 4, 4
A = np.random.rand(M * K).astype(dtype=np.float32).reshape(M, K)
X = np.random.rand(K * N).astype(dtype=np.float32).reshape(K, N)
################################################################################
#Process Grid Organization
#*************************
#
#The processes are arranged in a :math:`\sqrt{P} \times \sqrt{P}` grid, where :math:`P` is the total number of processes.
#
#Define
#
#.. math::
#   P' = \bigl \lceil \sqrt{P} \bigr \rceil
#
#and the replication factor
#
#.. math::
#   C = \bigl\lceil \tfrac{P}{P'} \bigr\rceil.
#
#Each process is assigned a pair of coordinates :math:`(g, l)` within this grid:
#
#.. math::
#   g = \mathrm{rank} \bmod P',
#   \quad
#   l = \left\lfloor \frac{\mathrm{rank}}{P'} \right\rfloor.
#
#For example, when :math:`P = 4` we have :math:`P' = 2`, giving a 2×2 layout:
#
#.. raw:: html
#
#   <div style="text-align: center; font-family: monospace; white-space: pre;">
#  ┌────────────┬────────────┐
#  │ Rank 0     │ Rank 1     │
#  │ (g=0, l=0) │ (g=1, l=0) │
#  ├────────────┼────────────┤
#  │ Rank 2     │ Rank 3     │
#  │ (g=0, l=1) │ (g=1, l=1) │
#  └────────────┴────────────┘
#   </div>

my_group = rank % p_prime
my_layer = rank // p_prime

# Create the sub‐communicators
layer_comm = comm.Split(color=my_layer, key=my_group)  # all procs in same layer
group_comm = comm.Split(color=my_group, key=my_layer)  # all procs in same group

blk_rows = int(math.ceil(M / p_prime))
blk_cols = int(math.ceil(N / p_prime))

rs = my_group * blk_rows
re = min(M, rs + blk_rows)
my_own_rows = re - rs

cs = my_layer * blk_cols
ce = min(N, cs + blk_cols)
my_own_cols = ce - cs

################################################################################
#Each rank will end up with:
#      - :math:`A_{p} \in \mathbb{R}^{\text{my_own_rows}\times K}`
#      - :math:`X_{p} \in \mathbb{R}^{K\times \text{my_own_cols}}`
#as follows:
A_p, X_p = A[rs:re, :].copy(), X[:, cs:ce].copy()

################################################################################
#.. raw:: html
#
#   <div style="text-align: left; font-family: monospace; white-space: pre;">
#   <b>Matrix A (4 x 4):</b>
#   ┌─────────────────┐
#   │ a11 a12 a13 a14 │ <- Rows 0–1 (Group 0)
#   │ a21 a22 a23 a24 │
#   ├─────────────────┤
#   │ a41 a42 a43 a44 │ <- Rows 2–3 (Group 1)
#   │ a51 a52 a53 a54 │
#   └─────────────────┘
#   </div>
#
#.. raw:: html
#
#   <div style="text-align: left; font-family: monospace; white-space: pre;">
#   <b>Matrix B (4 x 4):</b>
#   ┌─────────┬─────────┐
#   │ b11 b12 │ b13 b14 │ <- Cols 0–1 (Layer 0), Cols 2–3 (Layer 1)
#   │ b21 b22 │ b23 b24 │
#   │ b31 b32 │ b33 b34 │
#   │ b41 b42 │ b43 b44 │
#   └─────────┴─────────┘
#
#   </div>
#

################################################################################
#Forward Operation
#*****************
#To perform our distributed matrix-matrix multiplication :math:`Y = \text{Aop} \times X` we need to create our distributed operator :math:`\text{Aop}` and distributed operand :math:`X` from :math:`A_p` and
#:math:`X_p` respectively
Aop = MPIMatrixMult(A_p, N, dtype="float32")
################################################################################
# While as well passing the appropriate values.
col_lens = comm.allgather(my_own_cols)
total_cols =  np.sum(col_lens)
x = DistributedArray(global_shape=K * total_cols,
                     local_shapes=[K * col_len for col_len in col_lens],
                     partition=Partition.SCATTER,
                     mask=[i // p_prime for i in range(comm.Get_size())],
                     base_comm=comm,
                     dtype="float32")
x[:] = X_p.flatten()
################################################################################
#When we perform the matrix-matrix multiplication we shall then obtain a distributed :math:`Y` in the same way our :math:`X` was distributed.
y = Aop @ x
###############################################################################
#Adjoint Operation
#*****************
# In a similar fashion we then perform the Adjoint :math:`Xadj = A^H * Y`
xadj = Aop.H @ y
###############################################################################
#Here we verify the result against the equivalent serial version of the operation. Each rank checks that it has computed the correct values for it partition.
y_loc = A @ X
xadj_loc = (A.T.dot(y_loc.conj())).conj()

expected_y_loc = y_loc[:, cs:ce].flatten().astype(np.float32)
expected_xadj_loc = xadj_loc[:, cs:ce].flatten().astype(np.float32)

if not np.allclose(y.local_array, expected_y_loc, rtol=1e-6):
    print(f"RANK {rank}: FORWARD VERIFICATION FAILED")
    print(f'{rank} local: {y.local_array}, expected: {y_loc[:, cs:ce]}')
else:
    print(f"RANK {rank}: FORWARD VERIFICATION PASSED")

if not np.allclose(xadj.local_array, expected_xadj_loc, rtol=1e-6):
    print(f"RANK {rank}: ADJOINT VERIFICATION FAILED")
    print(f'{rank} local: {xadj.local_array}, expected: {xadj_loc[:, cs:ce]}')
else:
    print(f"RANK {rank}: ADJOINT VERIFICATION PASSED")

