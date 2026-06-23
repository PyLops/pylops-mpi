from typing import Tuple, Union

from pylops.utils.backend import get_module

from pylops_mpi.DistributedArray import DistributedArray, StackedDistributedArray
from pylops_mpi.LinearOperator import MPILinearOperator
from pylops_mpi.StackedLinearOperator import MPIStackedLinearOperator


def power_iteration(
        Op: Union[MPILinearOperator, MPIStackedLinearOperator],
        b_k: Union[DistributedArray, StackedDistributedArray],
        niter: int = 10,
        tol: float = 1e-5,
        dtype: str = "float64",
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
    Op : :obj:`pylops_mpi.MPILinearOperator` or :obj:`pylops_mpi.MPIStackedLinearOperator`
        Square operator
    b_k : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
        Initial guess for the eigenvector.
        This vector is modified in-place during the iteration and, upon convergence, contains the estimated
        dominant eigenvector (normalized to unit norm).
        The initial values of ``b_k`` are ignored, as it is re-initialized with random values inside the function.
        The operator ``Op`` must support distributed matrix-vector multiplication with ``b_k``
        (i.e., ``Op.matvec(b_k)``) consistent with the underlying partitioning.
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
    # Identify if operator is complex
    cmpx = 1j if ncp.issubdtype(ncp.dtype(dtype), ncp.complexfloating) else 0
    # Choose a random vector to decrease  the chance that vector is orthogonal to the eigenvector
    if isinstance(b_k, StackedDistributedArray):
        for iarr in range(b_k.narrays):
            dist = b_k[iarr]
            b_k[iarr][:] = (
                ncp.random.rand(dist.local_shape[0]).astype(dtype)
                + cmpx * ncp.random.rand(dist.local_shape[0]).astype(dtype)
            )
    else:
        b_k[:] = (
            ncp.random.rand(b_k.local_shape[0]).astype(dtype)
            + cmpx * ncp.random.rand(b_k.local_shape[0]).astype(dtype)
        )
    b_k_norm = b_k.norm()
    if isinstance(b_k, StackedDistributedArray):
        for iarr in range(b_k.narrays):
            b_k[iarr][:] /= b_k_norm
    else:
        b_k[:] /= b_k_norm
    maxeig_old = 0.0
    for iiter in range(niter):
        # Compute largest eigenvector
        b1_k = Op.matvec(b_k)

        # Compute largest eigenvalue
        maxeig = b_k.dot(b1_k, vdot=True).item()
        b1_k_norm = b1_k.norm()

        # Renormalize the vector
        if isinstance(b1_k, StackedDistributedArray):
            for iarr in range(b1_k.narrays):
                b_k[iarr][:] = b1_k[iarr][:] / b1_k_norm
        else:
            b_k[:] = b1_k[:] / b1_k_norm
        if ncp.abs(maxeig - maxeig_old) < tol * maxeig:
            break
        maxeig_old = maxeig
    return maxeig, b_k, iiter + 1
