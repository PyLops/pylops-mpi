r"""
Distributed Matrix Multiplication
=================================
This example shows how to use the :py:class:`pylops_mpi.basicoperators.MPIMatrixMult`
operator to perform matrix-matrix multiplication between a matrix :math:`\mathbf{A}`
blocked over rows (i.e., blocks of rows are stored over different ranks) and a
matrix :math:`\mathbf{X}` blocked over columns (i.e., blocks of columns are
stored over different ranks), with equal number of row and column blocks.
Similarly, the adjoint operation can be peformed with a matrix :math:`\mathbf{Y}`
blocked in the same fashion of matrix :math:`\mathbf{X}`.

Note that whilst the different blocks of the matrix :math:`\mathbf{A}` are directly
stored in the operator on different ranks, the matrix :math:`\mathbf{X}` is
effectively represented by a 1-D :py:class:`pylops_mpi.DistributedArray` where
the different blocks are flattened and stored on different ranks. Note that to
optimize communications, the ranks are organized in a 2D grid and some of the
row blocks of :math:`\mathbf{A}` and column blocks of :math:`\mathbf{X}` are
replicated across different ranks - see below for details.

"""

from matplotlib import pyplot as plt
import math
import numpy as np
from mpi4py import MPI

import pylops

import pylops_mpi
from pylops_mpi import Partition
from pylops_mpi.basicoperators.MatrixMult import active_grid_comm, MPIMatrixMult

plt.close("all")

###############################################################################
# We set the seed such that all processes can create the input matrices filled
# with the same random number. In practical application, such matrices will be
# filled with data that is appropriate that is appropriate the use-case.
np.random.seed(42)

###############################################################################
# We are now ready to create the input matrices :math:`\mathbf{A}` of size
# :math:`M \times k` :math:`\mathbf{A}` of size and :math:`\mathbf{A}` of size
# :math:`K \times N`.
N, K, M = 4, 4, 4
A = np.random.rand(N * K).astype(dtype=np.float32).reshape(N, K)
X = np.random.rand(K * M).astype(dtype=np.float32).reshape(K, M)

################################################################################
# The processes are now arranged in a :math:`P' \times P'` grid,
# where :math:`P` is the total number of processes.
#
# We define
#
# .. math::
#    P' = \bigl \lceil \sqrt{P} \bigr \rceil
#
# and the replication factor
#
# .. math::
#    R = \bigl\lceil \tfrac{P}{P'} \bigr\rceil.
#
# Each process is therefore assigned a pair of coordinates
# :math:`(r,c)` within this grid:
#
# .. math::
#    r = \left\lfloor \frac{\mathrm{rank}}{P'} \right\rfloor,
#    \quad
#    c = \mathrm{rank} \bmod P'.
#
# For example, when :math:`P = 4` we have :math:`P' = 2`, giving a 2×2 layout:
#
# .. raw:: html
#
#    <div style="text-align: center; font-family: monospace; white-space: pre;">
#   ┌────────────┬────────────┐
#   │ Rank 0     │ Rank 1     │
#   │ (r=0, c=0) │ (r=0, c=1) │
#   ├────────────┼────────────┤
#   │ Rank 2     │ Rank 3     │
#   │ (r=1, c=0) │ (r=1, c=1) │
#   └────────────┴────────────┘
#    </div>
#
# This is obtained by invoking the
# :func:`pylops_mpi.basicoperators.MPIMatrixMult.active_grid_comm` method, which is also
# responsible to identify any rank that should be deactivated (if the number
# of rows of the operator or columns of the input/output matrices are smaller
# than the row or columm ranks.

base_comm = MPI.COMM_WORLD
comm, rank, row_id, col_id, is_active = active_grid_comm(base_comm, N, M)
print(f"Process {base_comm.Get_rank()} is {'active' if is_active else 'inactive'}")
if not is_active: exit(0)

# Create sub‐communicators
p_prime = math.isqrt(comm.Get_size())
row_comm = comm.Split(color=row_id, key=col_id)  # all procs in same row
col_comm = comm.Split(color=col_id, key=row_id)  # all procs in same col

################################################################################
# At this point we divide the rows and columns of :math:`\mathbf{A}` and
# :math:`\mathbf{X}`, respectively, such that each rank ends up with:
#
#  - :math:`A_{p} \in \mathbb{R}^{\text{my_own_rows}\times K}`
#  - :math:`X_{p} \in \mathbb{R}^{K\times \text{my_own_cols}}`
#
# .. raw:: html
#
#   <div style="text-align: left; font-family: monospace; white-space: pre;">
#   <b>Matrix A (4 x 4):</b>
#   ┌─────────────────┐
#   │ a11 a12 a13 a14 │ <- Rows 0–1 (Process Grid Row 0)
#   │ a21 a22 a23 a24 │
#   ├─────────────────┤
#   │ a41 a42 a43 a44 │ <- Rows 2–3 (Process Grid Row 1)
#   │ a51 a52 a53 a54 │
#   └─────────────────┘
#   </div>
#
# .. raw:: html
#
#   <div style="text-align: left; font-family: monospace; white-space: pre;">
#   <b>Matrix X (4 x 4):</b>
#   ┌─────────┬─────────┐
#   │ b11 b12 │ b13 b14 │ <- Cols 0–1 (Process Grid Col 0), Cols 2–3 (Process Grid Col 1)
#   │ b21 b22 │ b23 b24 │
#   │ b31 b32 │ b33 b34 │
#   │ b41 b42 │ b43 b44 │
#   └─────────┴─────────┘
#   </div>
#

blk_rows = int(math.ceil(N / p_prime))
blk_cols = int(math.ceil(M / p_prime))

rs = col_id * blk_rows
re = min(N, rs + blk_rows)
my_own_rows = max(0, re - rs)

cs = row_id * blk_cols
ce = min(M, cs + blk_cols)
my_own_cols = max(0, ce - cs)

A_p, X_p = A[rs:re, :].copy(), X[:, cs:ce].copy()

################################################################################
# We are now ready to create the :py:class:`pylops_mpi.basicoperators.MPIMatrixMult`
# operator and the input matrix :math:`\mathbf{X}`
Aop = MPIMatrixMult(A_p, M, base_comm=comm, dtype="float32", kind="block")

col_lens = comm.allgather(my_own_cols)
total_cols = np.sum(col_lens)
x = pylops_mpi.DistributedArray(
    global_shape=K * total_cols,
    local_shapes=[K * col_len for col_len in col_lens],
    partition=Partition.SCATTER,
    mask=[i % p_prime for i in range(comm.Get_size())],
    base_comm=comm,
    dtype="float32")
x[:] = X_p.flatten()

################################################################################
# We can now apply the forward pass :math:`\mathbf{y} = \mathbf{Ax}` (which
# effectively implements a distributed matrix-matrix multiplication
# :math:`Y = \mathbf{AX}`). Note :math:`\mathbf{Y}` is distributed in the same
# way as the input :math:`\mathbf{X}`.
y = Aop @ x

###############################################################################
# Next we apply the adjoint pass :math:`\mathbf{x}_{adj} = \mathbf{A}^H \mathbf{x}`
# (which effectively implements a distributed matrix-matrix multiplication
# :math:`\mathbf{X}_{adj} = \mathbf{A}^H \mathbf{X}`). Note that
# :math:`\mathbf{X}_{adj}` is again distributed in the same way as the input
# :math:`\mathbf{X}`.
xadj = Aop.H @ y

###############################################################################
# Finally, we show the :py:class:`pylops_mpi.basicoperators.MPIMatrixMult`
# operator can be combined with any other PyLops-MPI operator. We are going to
# apply here a first derivative along the first axis to the output of the matrix
# multiplication. The only gotcha here is that one needs to be aware of the
# ad-hoc distribution of the arrays that are fed to this operator and make
# sure it is matched in the other operators involved in the chain.
Dop = pylops.FirstDerivative(dims=(N, my_own_cols), axis=0, 
                             dtype=np.float32)
DBop = pylops_mpi.MPIBlockDiag(ops=[Dop, ])
Op = DBop @ Aop

y1 = Op @ x