import warnings
from typing import Sequence

from mpi4py import MPI
import numpy as np

from pylops.signalprocessing._baseffts import _FFTNorms
from pylops.utils import DTypeLike, InputDimsLike

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
    fft : :obj:`mpi4py_fft.mpifft.PFFT`
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
    The MPIFFTND operator applies the forward and inverse N-dimensional FFT to a
    :class:`pylops_mpi.DistributedArray`, accepted as a 1D flattened array and reshaped internally
    to the layout defined by ``dims``. The distributed FFT transform is performed by
    :class:`mpi4py_fft.mpifft.PFFT` via :class:`mpi4py_fft.pencil.Subcomm`. Since the 1D input is
    always distributed along ``axis=0`` after reshaping, PFFT is configured to distribute along
    ``axis=0`` by default. The exception is when ``axes[-1] == 0``: PFFT requires the final
    transform axis to be local on each rank, so the distribution is shifted to ``axis=1`` and the
    input is redistributed accordingly before the transform. After the transform, the output is
    flattened back to 1D.

    The class uses PFFT's two internal pencil layouts: ``pencil[False]`` for forward-input/backward-output and
    ``pencil[True]`` for forward-output/backward-input. During initialization, it records the distributed axes
    of these layouts as ``_pfft_in_axis`` and ``_pfft_out_axis``, and redistributes the input
    :class:`pylops_mpi.DistributedArray` as needed before each transform.

    In the forward pass, :meth:`PFFT.forward` is called with ``normalize=False``, computing:

    .. math::
        D(k_1, \ldots, k_N) = \mathscr{F} (d) =
        \int\limits_{-\infty}^\infty \cdots \int\limits_{-\infty}^\infty
        d(x_1, \ldots, x_N)
        e^{-j2\pi k_1 x_1} \cdots
        e^{-j 2 \pi k_N x_N}  \,\mathrm{d}x_1 \cdots \mathrm{d}x_N

    When ``norm="1/n"``, the result is additionally scaled by :math:`1/N_F`.

    In the adjoint pass, :meth:`PFFT.backward` is called with ``normalize=True``, so ``PFFT``
    internally divides by :math:`N_F = \prod_i N_i`, computing:

    .. math::
        d(x_1, \ldots, x_N) = \mathscr{F}^{-1} (D) = \frac{1}{N_F}
        \int\limits_{-\infty}^\infty \cdots \int\limits_{-\infty}^\infty
        D(k_1, \ldots, k_N)
        e^{j2\pi k_1 x_1} \cdots
        e^{j 2 \pi k_N x_N} \,\mathrm{d}k_1 \cdots  \mathrm{d}k_N

    When ``norm="none"``, the adjoint multiplies by :math:`N_F` to cancel this internal scaling,
    returning a true unscaled adjoint. The result is then flattened back to a 1D
    :class:`pylops_mpi.DistributedArray`. All inter-rank data movement is handled internally by
    ``mpi4py_fft``.
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
        subcomm_dims = np.ones(len(dims), dtype=int)
        # axis=0 for the initial distribution by default
        # if the final axis over which FFT is applied is axis=0, the input array is first redistributed over axis=1
        # prior to applying FFT.
        if axes[-1] == 0:
            subcomm_dims[1] = 0
        else:
            subcomm_dims[0] = 0
        subcomm = Subcomm(base_comm, subcomm_dims)
        self.fft = PFFT(subcomm, self.dims, axes=self.axes, dtype=fft_dtype)

        # PFFT uses two internal layouts (pencils): one before and one after the transform. The two layouts can differ
        # because PFFT may redistribute data mid-transform to align the active FFT axis with the distributed axis.
        # pencil[False] is the forward-input / backward-output layout.
        # pencil[True] is the forward-output / backward-input layout.

        # Distributed axis in the pre-transform (pencil[False]) layout.
        self._pfft_in_axis = next(
            (i for i, s in enumerate(self.fft.pencil[False].subcomm) if s.Get_size() > 1), 0
        )
        # Distributed axis in the post-transform (pencil[True]) layout.
        self._pfft_out_axis = next(
            (i for i, s in enumerate(self.fft.pencil[True].subcomm) if s.Get_size() > 1), 0
        )

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
        # Redistribute input to match the input PFFT axis
        x = x.redistribute(axis=self._pfft_in_axis)
        x_dist_pfft[:] = x.local_array
        # Perform the parallel forward FFT
        self.fft.forward(x_dist_pfft, y_dist_pfft, normalize=False)

        y = DistributedArray(global_shape=self.dimsd, dtype=self.dtype, axis=self._pfft_out_axis,
                             base_comm=x.base_comm, base_comm_nccl=x.base_comm_nccl, engine=x.engine)
        y[:] = y_dist_pfft
        if self.real:
            self._scale_real_fft(y, inverse=False)
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
        if self.fftshift_after.any():
            x = ifftshift_nd(x, axes=self.axes[self.fftshift_after])
        if self.real:
            self._scale_real_fft(x, inverse=True)
        # Allocate distributed arrays for input and output
        y_dist_pfft = newDistArray(self.fft, forward_output=False)
        x_dist_pfft = newDistArray(self.fft, forward_output=True)
        # Redistribute input to match the PFFT axis
        x = x.redistribute(axis=self._pfft_out_axis)
        x_dist_pfft[:] = x.local_array
        # Perform the parallel backward FFT
        self.fft.backward(x_dist_pfft, y_dist_pfft, normalize=True)

        y = DistributedArray(global_shape=self.dims, dtype=self.dtype, axis=self._pfft_in_axis,
                             base_comm=x.base_comm, base_comm_nccl=x.base_comm_nccl, engine=x.engine)
        y[:] = y_dist_pfft
        if self.norm is _FFTNorms.NONE:
            y[:] *= self._scale
        if not self.clinear:
            y[:] = np.real(y.local_array)
        y[:] = y.local_array.astype(self.rdtype)
        if self.ifftshift_before.any():
            y = fftshift_nd(y, axes=self.axes[self.ifftshift_before])
        return y

    def _scale_real_fft(self, x: DistributedArray, inverse: bool = False) -> None:
        """Apply scaling for real-valued FFTs.

        Scales the non-DC positive frequency components along the final FFT axis
        by ``sqrt(2)`` in forward mode and ``1/sqrt(2)`` in inverse mode.

        When the final FFT axis is distributed across MPI ranks, only the local
        portion overlapping with the global positive-frequency range is scaled.

        Parameters
        ----------
        x : DistributedArray
            Distributed FFT array to scale in-place.
        inverse : bool, optional
            Apply inverse scaling when ``True``. Default is ``False``.
        """
        scale = 1 / np.sqrt(2) if inverse else np.sqrt(2)
        if x.axis == self.axes[-1]:
            sizes = [loc_shape[self.axes[-1]] for loc_shape in x.local_shapes]
            local_start = sum(sizes[:self.base_comm.rank])
            local_stop = local_start + sizes[self.base_comm.rank]
            freq_start, freq_stop = max(1, local_start), min(1 + (self.nffts[-1] - 1) // 2, local_stop)
            # Local overlap with the global frequency slice [1:k]
            if freq_stop > freq_start:
                local_slice = [slice(None)] * x.ndim
                local_slice[self.axes[-1]] = slice(freq_start - local_start, freq_stop - local_start)
                x[tuple(local_slice)] *= scale
        else:
            # Axis is local on this rank, so direct slicing
            freq_slice = [slice(None)] * x.ndim
            freq_slice[self.axes[-1]] = slice(1, 1 + (self.nffts[-1] - 1) // 2)
            x[tuple(freq_slice)] *= scale

    def __truediv__(self, y: DistributedArray) -> DistributedArray:
        y_div = self._rmatvec(y)
        y_div[:] = y_div.local_array / self._scale
        return y_div
