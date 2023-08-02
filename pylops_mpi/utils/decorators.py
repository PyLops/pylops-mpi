from functools import wraps
from typing import Callable, Optional

import numpy as np

from pylops_mpi import DistributedArray


def redistribute(
    func: Optional[Callable] = None,
) -> Callable:
    """Decorator used to reshape the model vector and flatten the data vector in a distributed fashion

    Parameters
    ----------
    func : :obj:`callable`, optional
        Function to be decorated

    Notes
    -----
    A ``_matvec`` (forward) function can be simplified to

    .. code-block:: python

        @redistribute
        def _matvec(self, x: DistributedArray):
            y = do_things_to_redistributed(y)
            return y

    where x will be reshaped to ``self.dims`` and y will be raveled to give a 1-D
    DistributedArray as output.

    """

    def decorator(f):
        @wraps(f)
        def wrapper(self, x: DistributedArray):
            arr = DistributedArray(global_shape=getattr(self, "dims"), axis=0, dtype=x.dtype)
            arr_local_shapes = np.asarray(arr.base_comm.allgather(np.prod(arr.local_shape)))
            x_local_shapes = np.asarray(x.base_comm.allgather(np.prod(x.local_shape)))
            # Calculate num_ghost_cells required for aeach rank
            dif = np.cumsum(arr_local_shapes - x_local_shapes)
            ghosted_array = x.add_ghost_cells(cells_back=dif[self.rank])
            # Fill the redistributed array
            arr[:] = ghosted_array[dif[self.rank - 1]:].reshape(arr.local_shape)
            y: DistributedArray = f(self, arr)
            y = y.ravel()
            return y
        return wrapper
    if func is not None:
        return decorator(func)
    return decorator
