from functools import wraps
from typing import Callable, Optional

import numpy as np
from mpi4py import MPI

from pylops_mpi import DistributedArray, Partition
from pylops_mpi.DistributedArray import local_split


def reshaped(
    func: Optional[Callable] = None,
    forward: Optional[bool] = None,
    stacking: Optional[bool] = None,
) -> Callable:
    """Decorator used to reshape the model vector and flatten the data vector in a distributed fashion.
    It is used in many operators.

    Parameters
    ----------
    func : :obj:`callable`, optional
        Function to be decorated.
    forward : :obj:`bool`, optional
        Mode of matrix-vector multiplication.
    stacking : :obj:`bool`, optional
        Set to ``True`` if it is a stacking operator.

    Notes
    -----
    A ``_matvec`` (forward) function can be simplified to

    .. code-block:: python

        @reshaped
        def _matvec(self, x: DistributedArray):
            y = do_things_to_redistributed(y)
            return y

    where x will be reshaped to ``self.dims`` and y will be raveled to give a 1-D
    DistributedArray as output.

    """

    def decorator(f):
        @wraps(f)
        def wrapper(self, x: DistributedArray):
            # Assign parameters for chosen path
            if x.partition is not Partition.SCATTER:
                raise ValueError(
                    f"x should have partition={Partition.SCATTER}, {x.partition} != {Partition.SCATTER}"
                )
            if stacking and forward:
                local_shapes = getattr(self, "local_shapes_m")
                global_shape = x.global_shape
            elif stacking and not forward:
                local_shapes = getattr(self, "local_shapes_n")
                global_shape = x.global_shape
            else:
                fwd = (
                    "rmat" not in f.__name__
                    and f.__name__ != "div"
                    and f.__name__ != "__truediv__"
                )
                local_shapes = None
                global_shape = (
                    getattr(self, "dims")
                    if fwd
                    else getattr(self, "dimsd", getattr(self, "dims"))
                )

            # Fast path: if x already has the shape and distribution that the operator expects
            # on every rank, pass it straight to f (no redistribution, hence no copies of x).
            # The decision is collective, since the slow path communicates.
            target_local_shape = (
                local_shapes[x.rank]
                if local_shapes
                else local_split(global_shape, x.base_comm, Partition.SCATTER, 0)
            )
            same = (
                tuple(x.global_shape) == tuple(global_shape)
                and x.axis == 0
                and x.mask is None
                and tuple(x.local_shape) == tuple(target_local_shape)
            )
            if x.base_comm.allreduce(int(same), op=MPI.MIN):
                y: DistributedArray = f(self, x)
                if len(y.global_shape) > 1:
                    y = y.redistribute(axis=0).ravel()
                return y

            # Slow path: x does not have the shape and/or distribution expected by the operator,
            # on at least for one rank. Create a new DistributedArray with the expected layout, exchange
            # the required elements between neighbouring ranks (via ghost cells), and fill it with
            # the (reshaped) local portion of x before passing it to f.
            arr = DistributedArray(
                global_shape=global_shape,
                base_comm=x.base_comm,
                base_comm_nccl=x.base_comm_nccl,
                local_shapes=local_shapes,
                axis=0,
                engine=x.engine,
                dtype=x.dtype,
            )
            arr_local_shapes = np.asarray(
                arr.base_comm.allgather(np.prod(arr.local_shape))
            )
            x_local_shapes = np.asarray(x.base_comm.allgather(np.prod(x.local_shape)))
            # Calculate num_ghost_cells required for each rank
            dif = np.cumsum(arr_local_shapes - x_local_shapes)
            # Calculate cells_front(0 means no ghost cells)
            cells_front = abs(min(0, dif[self.rank - 1]))
            # Calculate cells_back(0 means no ghost cells)
            cells_back = max(0, dif[self.rank])
            ghosted_array = x.add_ghost_cells(
                cells_front=cells_front, cells_back=cells_back
            )
            # Fill the redistributed array
            index = max(0, dif[self.rank - 1])
            arr[:] = ghosted_array[index : arr_local_shapes[self.rank] + index].reshape(
                arr.local_shape
            )
            y: DistributedArray = f(self, arr)
            if len(y.global_shape) > 1:
                # Make sure y is distributed along axis=0 before applying ravel
                y = y.redistribute(axis=0).ravel()
            return y

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
