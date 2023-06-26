import numpy as np
import logging
from typing import Optional, Union

from mpi4py import MPI

from pylops.utils import NDArray, DTypeLike, InputDimsLike
from pylops.utils._internal import _value_or_sized_to_array

from pylops_mpi import MPILinearOperator, DistributedArray, Partition

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class MatrixMult(MPILinearOperator):
    def __init__(
            self,
            A: NDArray,
            otherdims: Optional[Union[int, InputDimsLike]] = None,
            dtype: DTypeLike = "float64",
            base_comm: MPI.Comm = MPI.COMM_WORLD,
            kind: str = "all"
    ) -> None:
        self.A = A
        if isinstance(A, np.ndarray):
            self.complex = np.iscomplexobj(A)
        else:
            self.complex = np.iscomplexobj(A.data)
        if otherdims is None:
            dims, dimsd = (A.shape[1],), (A.shape[0],)
            self.reshape = False
            explicit = True
        else:
            otherdims = _value_or_sized_to_array(otherdims)
            self.otherdims = np.array(otherdims, dtype=int)
            dims, dimsd = np.insert(self.otherdims, 0, self.A.shape[1]), np.insert(
                self.otherdims, 0, self.A.shape[0]
            )
            self.dimsflatten, self.dimsdflatten = np.insert(
                [np.prod(self.otherdims)], 0, self.A.shape[1]
            ), np.insert([np.prod(self.otherdims)], 0, self.A.shape[0])
            self.reshape = True
            explicit = False
        self.shape = dimsd + dims
        # Check dtype for correctness (upcast to complex when A is complex)
        if np.iscomplexobj(A) and not np.iscomplexobj(np.ones(1, dtype=dtype)):
            dtype = A.dtype
            logging.warning("Matrix A is a complex object, dtype cast to %s" % dtype)
        super().__init__(
            dtype=np.dtype(dtype), shape=self.shape, base_comm=base_comm, kind=kind, explicit=explicit,
        )

    def _matvec(self, x):
        if self.kind in ("all", "force"):
            # To handle dot product
            x = DistributedArray.to_dist(x=x, partition=Partition.BROADCAST).local_array
        if self.reshape:
            x = np.reshape(x, self.dimsflatten)
        y = self.A.dot(x)
        if self.reshape:
            y = y.ravel()
        if self.kind == "master":
            return y
        return np.concatenate(self.base_comm.allgather(y))

    def _rmatvec(self, x):
        if self.kind in ("all", "force"):
            # To handle dot product
            x = DistributedArray.to_dist(x=x).local_array
        if self.reshape:
            x = np.reshape(x, self.dimsflatten)
        if self.complex:
            y = (self.A.T.dot(x.conj())).conj()
        else:
            y = self.A.T.dot(x)
        if self.reshape:
            return y.ravel()
        if self.kind == "master":
            return y
        return self.base_comm.allreduce(y)
