import numpy as np
from pylops.utils.backend import get_module
from pylops_mpi.DistributedArray import DistributedArray
from pylops_mpi.LinearOperator import MPILinearOperator
from typing import Tuple
from mpi4py import MPI


def power_iteration(
    Op: MPILinearOperator,
    niter: int = 10,
    tol: float = 1e-5,
    dtype: str = "float32",
    backend: str = "numpy",
) -> Tuple[float, DistributedArray, int]:
    """Power iteration algorithm.

    Power iteration algorithm, used to compute the largest eigenvector and
    corresponding eigenvalue. Note that for complex numbers, the eigenvalue
    with the largest module is found.

    This implementation closely follow that of
    https://en.wikipedia.org/wiki/Power_iteration.

    Parameters
    ----------
    Op : :obj:`pylops_mpi.MPILinearOperator`
        Square operator
    niter : :obj:`int`, optional
        Number of iterations
    tol : :obj:`float`, optional
        Update tolerance
    dtype : :obj:`str`, optional
        Type of elements in input array.
    backend : :obj:`str`, optional
        Backend to use (`numpy` or `cupy`)

    Returns
    -------
    maxeig : :obj:`float`
        Largest eigenvalue
    b_k : :obj:`pylops_mpi.DistributedArray`
        Largest eigenvector
    iiter : :obj:`int`
        Effective number of iterations

    """

    ncp = get_module(backend)
    cmpx = 1j if np.issubdtype(np.dtype(dtype), np.complexfloating) else 0
    b_k = DistributedArray(global_shape=Op.shape[1], dtype=dtype, engine=backend)
    b_k[:] = (
        ncp.random.rand(b_k.local_shape[0]).astype(dtype)
        + cmpx * ncp.random.rand(b_k.local_shape[0]).astype(dtype)
    )
    b_k[:] /= b_k.norm()
    maxeig_old = 0.0
    for iiter in range(niter):
        b1_k = Op.matvec(b_k)
        # Calculate vdot for the global array
        local_dot = ncp.vdot(b_k.local_array, b1_k.local_array)
        maxeig = b_k.base_comm.allreduce(local_dot, op=MPI.SUM)

        b_k[:] = b1_k.local_array / b1_k.norm()
        if ncp.abs(maxeig - maxeig_old) < tol * ncp.abs(maxeig):
            break
        maxeig_old = maxeig
    return maxeig, b_k, iiter + 1
