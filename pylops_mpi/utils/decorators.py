from functools import wraps
from typing import Callable, Optional

import numpy as np

from pylops_mpi import DistributedArray


def reshaped(
    func: Optional[Callable] = None,
) -> Callable:
    """Decorator used to reshape the model vector and flatten the data vector in a distributed fashion.
    It is used in many operators.

    Parameters
    ----------
    func : :obj:`callable`, optional
        Function to be decorated

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
            arr = DistributedArray(global_shape=getattr(self, "dims"), axis=0, dtype=x.dtype)
            arr_local_shape = np.prod(arr.local_shape)
            x_local_shape = np.prod(x.local_shape)
            if arr_local_shape != x_local_shape:
                raise ValueError(f"Dims {arr.local_shape} and shape of x {x.local_shape} doesn't align with "
                                 f"each other at rank={self.rank}; ({arr_local_shape}, ) != ({x_local_shape}, )")
            arr[:] = x.local_array.reshape(arr.local_shape)
            y: DistributedArray = f(self, arr)
            y_final = DistributedArray(global_shape=x.global_shape, dtype=y.dtype)
            y_final[:] = y.local_array.ravel()
            return y_final
        return wrapper
    if func is not None:
        return decorator(func)
    return decorator
