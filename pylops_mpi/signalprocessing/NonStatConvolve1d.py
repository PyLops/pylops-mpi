from typing import Union

import numpy as np

from mpi4py import MPI
from pylops.signalprocessing import NonStationaryConvolve1D
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

from pylops_mpi import MPILinearOperator
from pylops_mpi.basicoperators.BlockDiag import MPIBlockDiag
from pylops_mpi.basicoperators.Halo import MPIHalo, halo_block_split


def MPINonStationaryConvolve1D(
        dims: Union[int, InputDimsLike],
        hs: NDArray,
        ih: InputDimsLike,
        # axis: int = -1,
        base_comm: MPI.Comm = MPI.COMM_WORLD,
        dtype: DTypeLike = "float64",
    ) -> MPILinearOperator:
    r"""1D non-stationary convolution operator.

    Apply distributed non-stationary one-dimensional convolution.
    A varying compact filter is provided on a coarser grid and on-the-fly
    interpolation is applied in forward and adjoint modes. 

    Alongside distributing the input array across different ranks, the 
    filters are also distributed and filters operating at the edges of the
    local arrays are replicated on both ranks either side of the edge.

    .. note::

        Currently the 
        :obj:`pylops_mpi.signalprocessing.MPINonStationaryConvolve1D` 
        requires that shape of the local arrays of the input 
        :obj:`pylops_mpi.DistributedArray` are be the same for all ranks.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension of the input
        :obj:`pylops_mpi.DistributedArray`.
    hs : :obj:`numpy.ndarray`
        Bank of 1d compact filters of size :math:`n_\text{filts} \times n_h`.
        Filters must have odd number of samples and are assumed to be
        centered in the middle of the filter support.
    ih : :obj:`tuple`
        Indices of the locations of the filters ``hs`` in the model (and data). Note
        that the filters must be regularly sampled, i.e. :math:`dh=\text{diff}(ih)=\text{const.}`
    axis : :obj:`int`, optional
        Axis along which convolution is applied
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        MPI Base Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape

    Raises
    ------
    ValueError
        If filters ``hs`` have even size
    ValueError
        If ``ih`` is not regularly sampled
    ValueError
        If ``ih`` is outside the bounds defined by ``dims[axis]``

    Notes
    -----
    The MPINonStationaryConvolve1D operator applies non-stationary
    one-dimensional convolution between the input signal :math:`d(t)`
    and a bank of compact filter kernels :math:`h(t; t_i)`. Assume
    the input signal is composed of :math:`N=16` samples, and distributed
    across :math:`N=2` ranks (with each local signal composes of
    :math:`N=8` samples); similarly, consider :math:`N=4` filters
    at locations :math:`t_2` and :math:`t_6` in the first rank and
    :math:`t_10` and :math:`t_14` in the second rank. Each rank applies
    an halo of :math:`N=4` samples to include the following/preceding
    filter, and applies locally a
    :class:pylops.signalprocessingNonStationaryConvolve1D` operator;
    finally the halo is removed from each local convolved signal.

    """
    # TODO: need to adapt operator to handle NDarrays
    axis = -1
    rank = base_comm.Get_rank()
    size = base_comm.Get_size()
    dims = _value_or_sized_to_tuple(dims)

    # Checks for local operator
    if hs.shape[1] % 2 == 0:
        raise ValueError("filters hs must have odd length")
    if len(np.unique(np.diff(ih))) > 1:
        raise ValueError(
            "the indices of filters 'ih' are must be regularly sampled"
        )
    if min(ih) < 0 or max(ih) >= dims[axis]:
        raise ValueError(
            "the indices of filters 'ih' must be larger than 0 and "
            "smaller than `dims`"
        )

    # Additional check for distributed operarator
    if dims[axis] % size:
        raise ValueError(
            f"number of input samples {dims[0]} is not divisible by "
            f"the number of ranks ({size})"
        )

    # Identify halo as the maximum distance between any partition of
    # the distributed array into local arrays and any filter plus
    # half of the size of the filter
    dims_local = dims[axis] // size
    starts_local = np.arange(0, dims[axis], dims_local)
    start_local = starts_local[rank]
    end_local = start_local + dims_local - 1
    ihidx_local = np.where((ih >= start_local) & (ih <= end_local))[0]
    if len(ihidx_local) == 0:
        raise ValueError(f"rank {rank} has zerof filters!")
    ih_local = ih[ihidx_local]

    dist_start_local = 0 if rank == 0 else ih_local[0] - start_local
    dist_end_local = 0 if rank == (size - 1) else end_local - ih_local[-1]
    dist_start = np.max(np.array(base_comm.allgather(dist_start_local)))
    dist_end = np.max(np.array(base_comm.allgather(dist_end_local)))
    halo = max([dist_start, dist_end]) + (hs.shape[1] // 2 + hs.shape[1] % 2)

    # Create halo operator
    HOp = MPIHalo(
        dims=dims,
        halo=halo,
        proc_grid_shape=(size, ),
        comm=base_comm,
        dtype=dtype
    )

    x_slice = halo_block_split(dims, base_comm, (size, ))
    if rank == 0:
        COp = NonStationaryConvolve1D(
            dims=dims_local + halo, hs=hs[:ihidx_local[-1] + 2],
            ih=ih[:ihidx_local[-1] + 2],
            dtype=dtype
        )
    elif rank == size - 1:
        COp = NonStationaryConvolve1D(
            dims=dims_local + halo, hs=hs[ihidx_local[0] - 1:],
            ih=ih[ihidx_local[0] - 1:] - x_slice[0].start + halo,
            dtype=dtype
        )
    else:
        COp = NonStationaryConvolve1D(
            dims=dims_local + 2 * halo, hs=hs[ihidx_local[0] - 1: ihidx_local[-1] + 2],
            ih=ih[ihidx_local[0] - 1: ihidx_local[-1] + 2] - x_slice[0].start + halo,
            dtype=dtype
        )

    COp_full = MPIBlockDiag([COp, ])
    Op = HOp.H @ COp_full @ HOp

    return Op
