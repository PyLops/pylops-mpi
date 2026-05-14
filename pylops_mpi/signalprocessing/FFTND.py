import warnings
from typing import Sequence

from mpi4py import MPI
import numpy as np

from pylops.signalprocessing._baseffts import _FFTNorms
from pylops.utils import DTypeLike, InputDimsLike, get_array_module

from pylops_mpi.utils.decorators import reshaped
from pylops_mpi.DistributedArray import DistributedArray, Partition
from pylops_mpi.signalprocessing._baseffts import _MPIBaseFFTND
from pylops_mpi.utils import deps, fftshift_nd, ifftshift_nd

mpi4py_fft_message = deps.mpi4py_fft_import("mpi4py_fft")

if mpi4py_fft_message is None:
    from mpi4py_fft import PFFT, newDistArray
    from mpi4py_fft.pencil import Subcomm


class MPIFFTND(_MPIBaseFFTND):
    r"""N-dimensional Fast-Fourier Transform.

    Apply N-dimensional Fast-Fourier Transform (FFT) to any n ``axes``
    of a multidimensional array.

    When using ``real=True``, the result of the forward is also multiplied by
    :math:`\sqrt{2}` for all frequency bins except zero and Nyquist along the last
    ``axes``, and the input of the adjoint is multiplied by
    :math:`1 / \sqrt{2}` for the same frequencies.

    For a real valued input signal, it is advised to use the flag ``real=True``
    as it stores the values of the Fourier transform of the last axis in ``axes`` at positive
    frequencies only as values at negative frequencies are simply their complex conjugates.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    axes : :obj:`tuple`, optional
        Axes (or axis) along which FFTND is applied
    sampling : :obj:`tuple` or :obj:`float`, optional
        Sampling steps for each direction. When supplied a single value, it is used
        for all directions.
    norm : `{"none", "1/n"}`, optional
        - "none": Does not scale the forward or the adjoint FFT transforms. Default is "none".
        - "1/n": Scales both the forward and adjoint FFT transforms by
          :math:`1/N_F`.
    real : :obj:`bool`, optional
        Model to which fft is applied has real numbers (``True``) or not
        (``False``). Used to enforce that the output of adjoint of a real
        model is real. Note that the real FFT is applied only to the first
        dimension to which the FFTND operator is applied (last element of
        ``axes``)
    ifftshift_before : :obj:`tuple` or :obj:`bool`, optional
        Apply ifftshift (``True``) or not (``False``) to model vector (before FFT).
        Consider using this option when the model vector's respective axis is symmetric
        with respect to the zero value sample. This will shift the zero value sample to
        coincide with the zero index sample. With such an arrangement, FFT will not
        introduce a sample-dependent phase-shift when compared to the continuous Fourier
        Transform. When passing a single value, the shift will the same for every direction.
        Pass a tuple to specify which dimensions are shifted.
    fftshift_after : :obj:`tuple` or :obj:`bool`, optional
        Apply fftshift (``True``) or not (``False``) to data vector (after FFT).
        Consider using this option when you require frequencies to be arranged
        naturally, from negative to positive. When not applying fftshift after FFT,
        frequencies are arranged from zero to largest positive, and then from negative
        Nyquist to the frequency bin before zero. When passing a single value, the shift
        will the same for every direction. Pass a tuple to specify which dimensions are shifted.
    dtype : :obj:`str`, optional
        Type of elements in input array. Note that the ``dtype`` of the operator
        is the corresponding complex type even when a real type is provided.
        In addition, note that the NumPy backend does not support returning ``dtype``
        different from ``complex128``.
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        MPI Base Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.

    Attributes
    ----------
    fs : :obj:`tuple`
        Each element of the tuple corresponds to the Discrete Fourier Transform
        sample frequencies along the respective direction given by ``axes``.
    nffts : :obj:`tuple` or :obj:`int`, optional
        Number of samples in Fourier Transform for each axis in ``axes``.
    real : :obj:`bool`
        When ``True``, uses real fast fourier transform
    rdtype : :obj:`bool`
        Expected input type to the forward
    cdtype : :obj:`bool`
        Output type of the forward. Complex equivalent to ``rdtype``.
    shape : :obj:`tuple`
        Operator shape.
    clinear : :obj:`bool`
        Operator is complex-linear. Is false when either ``real=True`` or when
        ``dtype`` is not a complex type.
    fft : :obj:`mpi4py_fft.PFFT`
        Parallel FFT operator object handling the distributed transform across
        MPI processes. Configured with the base communicator, dimension
        decomposition, transform axes, and dtype.

    See Also
    --------
    MPIFFT2D: Two-dimensional FFT

    Raises
    ------
    ValueError
        - If ``norm`` is not one of "none", or "1/n".

    Notes
    -----
    The MPIFFTND operator (using ``norm="none"``) applies the N-dimensional forward
    Fourier transform to a multidimensional array. Considering an N-dimensional
    signal :math:`d(x_1, \ldots, x_N)`. The MPIFFTND in forward mode is:

    .. math::
        D(k_1, \ldots, k_N) = \mathscr{F} (d) =
        \int\limits_{-\infty}^\infty \cdots \int\limits_{-\infty}^\infty
        d(x_1, \ldots, x_N)
        e^{-j2\pi k_1 x_1} \cdots
        e^{-j 2 \pi k_N x_N}  \,\mathrm{d}x_1 \cdots \mathrm{d}x_N

    Similarly, the N-dimensional inverse Fourier transform is applied to
    the Fourier spectrum :math:`D(k_1, \ldots, k_N)` in adjoint mode:

    .. math::
        d(x_1, \ldots, x_N) = \mathscr{F}^{-1} (D) = \frac{1}{N_F}
        \int\limits_{-\infty}^\infty \cdots \int\limits_{-\infty}^\infty
        D(k_1, \ldots, k_N)
        e^{j2\pi k_1 x_1} \cdots
        e^{j 2 \pi k_N x_N} \,\mathrm{d}k_1 \cdots  \mathrm{d}k_N

    where :math:`N_F` is the number of samples in the Fourier domain given by the
    product of the elements of ``nffts``.

    Both operators are effectively discretized and solved by a fast iterative
    algorithm known as Fast Fourier Transform. Note that when using ``norm="none"``,
    the adjoint is **not** the inverse of the forward mode; instead, the inverse
    requires an explicit :math:`1/N_F` scaling factor (applied in the adjoint/inverse).

    """
    def __init__(
        self,
        dims: InputDimsLike,
        axes: InputDimsLike = (0, 1, 2),
        sampling: float | Sequence[float] = 1.0,
        norm: str = "none",
        real: bool = False,
        ifftshift_before: bool = False,
        fftshift_after: bool = False,
        dtype: DTypeLike = "complex128",
        base_comm: MPI.Comm = MPI.COMM_WORLD
    ) -> None:
        super().__init__(
            dims=dims,
            axes=axes,
            sampling=sampling,
            norm=norm,
            real=real,
            fftshift_after=fftshift_after,
            ifftshift_before=ifftshift_before,
            dtype=dtype,
            base_comm=base_comm
        )
        if self.cdtype != np.complex128:
            warnings.warn(
                "numpy backend always returns complex128 dtype. "
                "To respect the passed dtype, data will be cast to {self.cdtype}.",
                stacklevel=2,
            )
        if self.norm is _FFTNorms.NONE:
            self._scale = np.prod(self.nffts)
        elif self.norm is _FFTNorms.ONE_OVER_N:
            self._scale = 1.0 / np.prod(self.nffts)
        fft_dtype = self.rdtype if self.real else self.cdtype
        # Distribute only along axes[0]; all other axes are non-distributed (0=distributed, 1=not)
        subcomm_dims = np.ones(len(dims), dtype=int)
        subcomm_dims[axes[0]] = 0
        subcomm = Subcomm(base_comm, subcomm_dims)
        self.fft = PFFT(subcomm, self.dims, axes=self.axes, dtype=fft_dtype, collapse=True)

    @reshaped
    def _matvec(self, x: DistributedArray) -> DistributedArray:
        if x.engine == "cupy":
            raise ValueError(f"x should be a numpy array with engine=numpy"
                             f"Got  {x.engine} instead...")
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER}"
                             f"Got  {x.partition} instead...")
        if self.ifftshift_before.any():
            x = ifftshift_nd(x, axes=self.axes[self.ifftshift_before])
        if not self.clinear:
            x[:] = np.real(x.local_array)
        x_dist_pfft = newDistArray(self.fft, forward_output=False)
        y_dist_pfft = newDistArray(self.fft, forward_output=True)
        # Redistribute input to match the axis decomposed by PFFT
        x = x.redistribute(axis=self.axes[0])
        x_dist_pfft[:] = x.local_array
        # Perform the parallel forward FFT
        self.fft.forward(x_dist_pfft, y_dist_pfft, normalize=False)

        # Axis along which PFFT decomposes the output array across MPI processes
        dist_axis = [i for i, s in enumerate(y_dist_pfft.subcomm) if s.Get_size() > 1]
        y = DistributedArray(global_shape=self.dimsd, dtype=self.dtype, axis=dist_axis[0] if dist_axis else 0,
                             base_comm=x.base_comm, base_comm_nccl=x.base_comm_nccl, engine=x.engine)
        y[:] = y_dist_pfft
        if self.real:
            if y.axis == self.axes[-1]:
                sizes = [loc_shape[self.axes[-1]] for loc_shape in y.local_shapes]
                start = sum(sizes[:self.base_comm.rank])
                stop = start + sizes[self.base_comm.rank]
                # Local overlap with the global frequency slice [1:k]
                g0, g1 = max(1, start), min(1 + (self.nffts[-1] - 1) // 2, stop)
                if g1 > g0:
                    sl = [slice(None)] * y.ndim
                    sl[self.axes[-1]] = slice(g0 - start, g1 - start)
                    y[tuple(sl)] *= np.sqrt(2)
            else:
                # Axis is local on this rank, so direct slicing
                sl = [slice(None)] * y.ndim
                sl[self.axes[-1]] = slice(1, 1 + (self.nffts[-1] - 1) // 2)
                y[tuple(sl)] *= np.sqrt(2)
        if self.norm is _FFTNorms.ONE_OVER_N:
            y[:] *= self._scale
        y[:] = y.local_array.astype(self.cdtype)
        if self.fftshift_after.any():
            y = fftshift_nd(y, axes=self.axes[self.fftshift_after])
        return y

    @reshaped
    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        if x.engine == "cupy":
            raise ValueError(f"x should be a numpy array with engine=numpy"
                             f"Got  {x.engine} instead...")
        if x.partition != Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER}, "
                             f"Got  {x.partition} instead...")
        np = get_array_module(x.local_array)
        if self.fftshift_after.any():
            x = ifftshift_nd(x, axes=self.axes[self.fftshift_after])
        if self.real:
            if x.axis == self.axes[-1]:
                sizes = [loc_shape[self.axes[-1]] for loc_shape in x.local_shapes]
                start = sum(sizes[:self.base_comm.rank])
                stop = start + sizes[self.base_comm.rank]
                # Local overlap with the global frequency slice [1:k]
                g0, g1 = max(1, start), min(1 + (self.nffts[-1] - 1) // 2, stop)
                if g1 > g0:
                    sl = [slice(None)] * x.ndim
                    sl[self.axes[-1]] = slice(g0 - start, g1 - start)
                    x[tuple(sl)] /= np.sqrt(2)
            else:
                # Axis is local on this rank, so direct slicing
                sl = [slice(None)] * x.ndim
                sl[self.axes[-1]] = slice(1, 1 + (self.nffts[-1] - 1) // 2)
                x[tuple(sl)] /= np.sqrt(2)
        # Allocate distributed arrays for input and output
        y_dist_pfft = newDistArray(self.fft, forward_output=False)
        x_dist_pfft = newDistArray(self.fft, forward_output=True)
        # Redistribute input to match the axis decomposed by PFFT
        x_axis = [i for i, s in enumerate(x_dist_pfft.subcomm) if s.Get_size() > 1]
        x = x.redistribute(axis=x_axis[0] if x_axis else 0)
        x_dist_pfft[:] = x.local_array
        # Perform the parallel backward FFT
        self.fft.backward(x_dist_pfft, y_dist_pfft, normalize=True)

        # Axis along which PFFT decomposes the output array across MPI processes
        dist_axis = [i for i, s in enumerate(y_dist_pfft.subcomm) if s.Get_size() > 1]
        y = DistributedArray(global_shape=self.dims, dtype=self.dtype, axis=dist_axis[0] if dist_axis else 0,
                             base_comm=x.base_comm, base_comm_nccl=x.base_comm_nccl, engine=x.engine)
        y[:] = y_dist_pfft
        if self.norm is _FFTNorms.NONE:
            y[:] *= self._scale
        if self.nffts[0] > self.dims[self.axes[0]]:
            y[:] = np.take(y.local_array, np.arange(self.dims[self.axes[0]]), axis=self.axes[0])
        if self.nffts[1] > self.dims[self.axes[1]]:
            y[:] = np.take(y.local_array, np.arange(self.dims[self.axes[1]]), axis=self.axes[1])
        if not self.clinear:
            y[:] = np.real(y.local_array)
        y[:] = y.local_array.astype(self.rdtype)
        if self.ifftshift_before.any():
            y = fftshift_nd(y, axes=self.axes[self.ifftshift_before])
        return y

    def __truediv__(self, y: DistributedArray) -> DistributedArray:
        y_div = self._rmatvec(y)
        y_div[:] = y_div.local_array / self._scale
        return y_div
