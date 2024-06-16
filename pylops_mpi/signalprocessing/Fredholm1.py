import numpy as np

from mpi4py import MPI
from pylops.utils.backend import get_module
from pylops.utils.typing import DTypeLike, NDArray

from pylops_mpi import (
    DistributedArray,
    MPILinearOperator,
    Partition
)

from pylops_mpi.utils.decorators import reshaped


class MPIFredholm1(MPILinearOperator):
    r"""Fredholm integral of first kind.

    Implement a multi-dimensional Fredholm integral of first kind distributed
    across the first dimension

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel of size
        :math:`[n_{\text{slice}} \times n_x \times n_y]`
    nz : :obj:`int`, optional
        Additional dimension of model
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G.H`` to speed up the computation of adjoint
        (``True``) or create ``G.H`` on-the-fly (``False``)
        Note that ``saveGt=True`` will double the amount of required memory
    usematmul : :obj:`bool`, optional
        Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
        (``False``). As it is not possible to define which approach is more
        performant (this is highly dependent on the size of ``G`` and input
        arrays as well as the hardware used in the computation), we advise users
        to time both methods for their specific problem prior to making a
        choice.
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
    NotImplementedError
        If the size of the first dimension of ``G`` is equal to 1 in any of the ranks
    
    Notes
    -----
    A multi-dimensional Fredholm integral of first kind can be expressed as

    .. math::

        d(k, x, z) = \int{G(k, x, y) m(k, y, z) \,\mathrm{d}y}
        \quad \forall k=1,\ldots,n_{slice}

    on the other hand its adjoint is expressed as

    .. math::

        m(k, y, z) = \int{G^*(k, y, x) d(k, x, z) \,\mathrm{d}x}
        \quad \forall k=1,\ldots,n_{\text{slice}}

    This integral is implemented in a distributed fashion, where ``G`` 
    is split across ranks along its first dimension. The inputs
    of both the forward and adjoint are distributed arrays with broadcast partion:
    each rank takes a portion of such arrays, computes a partial integral, and
    the resulting outputs are then gathered by all ranks to return a
    distributed arrays with broadcast partion.

    """

    def __init__(
        self,
        G: NDArray,
        nz: int = 1,
        saveGt: bool = False,
        usematmul: bool = True,
        base_comm: MPI.Comm = MPI.COMM_WORLD,
        dtype: DTypeLike = "float64",
    ) -> None:
        self.nz = nz
        self.nsl, self.nx, self.ny = G.shape
        self.nsls = base_comm.allgather(self.nsl)
        if base_comm.Get_rank() == 0 and 1 in self.nsls:
            raise NotImplementedError(f'All ranks must have at least 2 or more '
                                      f'elements in the first dimension: '
                                      f'local split is instead {self.nsls}...')
        nslstot = base_comm.allreduce(self.nsl)
        self.islstart = np.insert(np.cumsum(self.nsls)[:-1], 0, 0)
        self.islend = np.cumsum(self.nsls)
        self.rank = base_comm.Get_rank()
        self.dims = (nslstot, self.ny, self.nz)
        self.dimsd = (nslstot, self.nx, self.nz)
        shape = (np.prod(self.dimsd), 
                 np.prod(self.dims))
        super().__init__(shape=shape, dtype=np.dtype(dtype), base_comm=base_comm)

        self.G = G
        if saveGt:
            self.GT = G.transpose((0, 2, 1)).conj()
        self.usematmul = usematmul

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition is not Partition.BROADCAST:
            raise ValueError(f"x should have partition={Partition.BROADCAST}, {x.partition} != {Partition.BROADCAST}")
        y = DistributedArray(global_shape=self.shape[0], partition=Partition.BROADCAST,
                             engine=x.engine, dtype=self.dtype)
        x = x.local_array.reshape(self.dims).squeeze()
        x = x[self.islstart[self.rank]:self.islend[self.rank]]
        # apply matmul for portion of the rank of interest
        if self.usematmul:
            if self.nz == 1:
                x = x[..., ncp.newaxis]
            y1 = ncp.matmul(self.G, x)
        else:
            y1 = ncp.squeeze(ncp.zeros((self.nsls[self.rank], self.nx, self.nz), dtype=self.dtype))
            for isl in range(self.nsls[self.rank]):
                y1[isl] = ncp.dot(self.G[isl], x[isl])
        # gather results
        y[:] = np.vstack(self.base_comm.allgather(y1)).ravel()
        return y

    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_module(x.engine)
        if x.partition is not Partition.BROADCAST:
            raise ValueError(f"x should have partition={Partition.BROADCAST}, {x.partition} != {Partition.BROADCAST}")
        y = DistributedArray(global_shape=self.shape[1], partition=Partition.BROADCAST,
                             engine=x.engine, dtype=self.dtype)
        x = x.local_array.reshape(self.dimsd).squeeze()
        x = x[self.islstart[self.rank]:self.islend[self.rank]]
        # apply matmul for portion of the rank of interest
        if self.usematmul:
            if self.nz == 1:
                x = x[..., ncp.newaxis]
            if hasattr(self, "GT"):
                y1 = ncp.matmul(self.GT, x)
            else:
                y1 = (
                    ncp.matmul(x.transpose(0, 2, 1).conj(), self.G)
                    .transpose(0, 2, 1)
                    .conj()
                )
        else:
            y1 = ncp.squeeze(ncp.zeros((self.nsls[self.rank], self.ny, self.nz), dtype=self.dtype))
            if hasattr(self, "GT"):
                for isl in range(self.nsls[self.rank]):
                    y1[isl] = ncp.dot(self.GT[isl], x[isl])
            else:
                for isl in range(self.nsl):
                    y1[isl] = ncp.dot(x[isl].T.conj(), self.G[isl]).T.conj()
        
        # gather results
        y[:] = np.vstack(self.base_comm.allgather(y1)).ravel()
        return y
        