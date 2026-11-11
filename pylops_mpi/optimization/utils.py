__all__ = ["first_distarray"]

from pylops_mpi import DistributedArray, StackedDistributedArray


def first_distarray(
    x: DistributedArray | StackedDistributedArray,
) -> DistributedArray:
    """Return the first :class:`pylops_mpi.DistributedArray` of ``x``

    Used by solvers to display the first element of a model. Nested
    :class:`pylops_mpi.StackedDistributedArray` objects are unwrapped
    recursively.

    Parameters
    ----------
    x : :obj:`pylops_mpi.DistributedArray` or :obj:`pylops_mpi.StackedDistributedArray`
        Distributed array, possibly (nested) stacked

    Returns
    -------
    x : :obj:`pylops_mpi.DistributedArray`
        ``x`` itself if it is a :class:`pylops_mpi.DistributedArray`,
        otherwise its first (innermost) :class:`pylops_mpi.DistributedArray`

    """
    while isinstance(x, StackedDistributedArray):
        x = x.distarrays[0]
    return x
