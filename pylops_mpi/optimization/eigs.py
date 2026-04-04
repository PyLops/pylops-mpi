from mpi4py import MPI
from pylops.utils.backend import get_module
from pylops_mpi.DistributedArray import DistributedArray, NcclCommunicatorType
from pylops_mpi.LinearOperator import MPILinearOperator
from typing import Tuple, Optional


def power_iteration(
    Op: MPILinearOperator,
    niter: int = 10,
    tol: float = 1e-5,
    base_comm: Optional[MPI.Comm] = MPI.COMM_WORLD,
    base_comm_nccl: Optional[NcclCommunicatorType] = None,
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
    base_comm : :obj:`MPI.Comm`
        Base MPI Communicator.
    base_comm_nccl : `cupy.cuda.nccl.NcclCommunicator`
        NCCL Communicator
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
    cmpx = 1j if ncp.issubdtype(ncp.dtype(dtype), ncp.complexfloating) else 0
    b_k = DistributedArray(global_shape=Op.shape[1], dtype=dtype, engine=backend,
                           base_comm=base_comm, base_comm_nccl=base_comm_nccl)
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
        maxeig = b_k._allreduce(b_k.base_comm, b_k.base_comm_nccl, local_dot, engine=backend).item()
        b_k[:] = b1_k.local_array / b1_k.norm()
        if ncp.abs(maxeig - maxeig_old) < tol * ncp.abs(maxeig):
            break
        maxeig_old = maxeig
    return maxeig, b_k, iiter + 1
