all = [
    "fftshift",
    "ifftshift"
]

import numpy as np

from pylops.utils import InputDimsLike

from pylops_mpi import DistributedArray


def fftshift(x: DistributedArray, axes: InputDimsLike = None):
    """
    Shift the zero-frequency component to the center of the spectrum for a DistributedArray.

    This is the distributed equivalent of :func:`numpy.fft.fftshift`. For axes
    that are local to each process, the shift is applied directly. For the
    distributed axis, the array is first redistributed to a different axis so
    the shift can be performed locally, then left in the redistributed state.

    Parameters
    ----------
    x : :obj: `pylops_mpi.DistributedArray`
        Input array to shift. Modified in-place along each axis.
    axes : tuple, optional
        Axes over which to shift. Defaults to all axes if not specified.

    Returns
    -------
    x : :obj: `pylops_mpi.DistributedArray`
        The shifted array. May be distributed along a different axis than the
        input if the original distributed axis was included in ``axes``.
    """
    if axes is None:
        axes = tuple(range(x.ndim))
    elif np.isscalar(axes):
        axes = (axes,)
    local_axes = [ax for ax in axes if ax != x.axis]
    remote_axes = [ax for ax in axes if ax == x.axis]
    if local_axes:
        shifts = [x.global_shape[ax] // 2 for ax in local_axes]
        x[:] = np.roll(x.local_array, shift=shifts, axis=local_axes)
    if remote_axes:
        new_axis = 1 if x.axis == 0 else 0
        # Redistribute to a new axis for computation
        x = x.redistribute(axis=new_axis)
        shifts = [x.global_shape[ax] // 2 for ax in remote_axes]
        x[:] = np.roll(x.local_array, shift=shifts, axis=remote_axes)
    return x


def ifftshift(x: DistributedArray, axes: InputDimsLike = None):
    """
    Shift the zero-frequency component back to the beginning of the spectrum for a DistributedArray.

    This is the distributed equivalent of :func:`numpy.fft.ifftshift``.
    Shifts are applied in the negative direction (i.e. ``-(n // 2)`` per axis) to undo
    a prior :func:`pylops_mpi.utils.fftshift`. For axes that are local to each process, the shift is applied directly.
    For the distributed axis, the array is first redistributed to a different
    axis so the shift can be performed locally.

    Parameters
    ----------
    x : :obj: `pylops_mpi.DistributedArray`
        Input array to shift. Modified in-place along each axis.
    axes : int or sequence of int, optional
        Axes over which to shift. Defaults to all axes if not specified.

    Returns
    -------
    x : :obj: `pylops_mpi.DistributedArray`
        The shifted array. May be distributed along a different axis than the
        input if the original distributed axis was included in ``axes``.
    """
    if axes is None:
        axes = tuple(range(x.ndim))
    elif np.isscalar(axes):
        axes = (axes,)
    local_axes = [ax for ax in axes if ax != x.axis]
    dist_axes = [ax for ax in axes if ax == x.axis]
    if local_axes:
        shifts = [-(x.global_shape[ax] // 2) for ax in local_axes]
        x[:] = np.roll(x.local_array, shift=shifts, axis=local_axes)
    if dist_axes:
        new_axis = 1 if x.axis == 0 else 0
        # Redistribute to a new axis for computation
        x = x.redistribute(axis=new_axis)
        shifts = [-(x.global_shape[ax] // 2) for ax in dist_axes]
        x[:] = np.roll(x.local_array, shift=shifts, axis=dist_axes)
    return x
