r"""
Distributed SUMMA Matrix Multiplication
=======================================
This example shows how to use the :py:class:`pylops_mpi.basicoperators.MPISummaMatrixMult`
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

import pylops_mpi
from pylops_mpi.basicoperators.MatrixMult import (local_block_spit, block_gather, MPIMatrixMult)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 9
M = 9
K = 9

A_shape = (N, K)
x_shape = (K, M)
y_shape = (N, M)

p_prime = math.isqrt(size)
A_data = np.arange(int(A_shape[0] * A_shape[1])).reshape(A_shape)
x_data = np.arange(int(x_shape[0] * x_shape[1])).reshape(x_shape)

A_slice = local_block_spit(A_shape, rank, comm)
x_slice = local_block_spit(x_shape, rank, comm)
A_local = A_data[A_slice]
x_local = x_data[x_slice]

x_dist = pylops_mpi.DistributedArray(global_shape=(K * M),
                                     local_shapes=comm.allgather(x_local.shape[0] * x_local.shape[1]),
                                     base_comm=comm,
                                     partition=pylops_mpi.Partition.SCATTER,
                                     dtype=x_local.dtype)
x_dist.local_array[:] = x_local.flatten()

Aop = MPIMatrixMult(A_local, M, base_comm=comm, kind="summa", dtype=A_local.dtype)
y_dist = Aop @ x_dist
xadj_dist = Aop.H @ y_dist

y = block_gather(y_dist, (N,M), (N,M), comm)
xadj = block_gather(xadj_dist, (K,M), (K,M), comm)
if rank == 0 :
    y_correct = np.allclose(A_data @ x_data, y)
    print("y expected: ", y_correct)
    if not y_correct:
        print("expected:\n", A_data @ x_data)
        print("calculated:\n",y)

    xadj_correct = np.allclose((A_data.T.dot((A_data @ x_data).conj())).conj(), xadj.astype(np.int32))
    print("xadj expected: ", xadj_correct)
    if not xadj_correct:
        print("expected:\n", (A_data.T.dot((A_data @ x_data).conj())).conj())
        print("calculated:\n", xadj.astype(np.int32))
