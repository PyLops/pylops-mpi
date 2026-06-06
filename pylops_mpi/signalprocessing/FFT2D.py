from typing import Sequence

from mpi4py import MPI

from pylops.utils import DTypeLike, InputDimsLike

from pylops_mpi.DistributedArray import DistributedArray
from pylops_mpi.signalprocessing.FFTND import MPIFFTND


class MPIFFT2D(MPIFFTND):
    r"""Two-dimensional Fast-Fourier Transform.

    Apply two-dimensional Fast-Fourier Transform (FFT) to any pair of ``axes`` of a
    multidimensional array.

    When using ``real=True``, the result of the forward is also multiplied by
    :math:`\sqrt{2}` for all frequency bins except zero and Nyquist, and the input of
    the adjoint is multiplied by :math:`1 / \sqrt{2}` for the same frequencies.

    For a real valued input signal, it is advised to use the flag ``real=True``
    as it stores the values of the Fourier transform of the last axis in ``axes`` at positive
    frequencies only as values at negative frequencies are simply their complex conjugates.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    axes : :obj:`tuple`, optional
        Pair of axes along which FFT2D is applied
    sampling : :obj:`tuple` or :obj:`float`, optional
        Sampling steps for each axis in ``axes``. When supplied a single value, it is used
        for both axes.
    norm : `{"none", "1/n"}`, optional
        - "none": Does not scale the forward or the adjoint FFT transforms. Default is "none".
        - "1/n": Scales both the forward and adjoint FFT transforms by
          :math:`1/N_F`.
    real : :obj:`bool`, optional
        Model to which fft is applied has real numbers (``True``) or not
        (``False``). Used to enforce that the output of adjoint of a real
        model is real. Note that the real FFT is applied only to the first
        dimension to which the FFT2D operator is applied (last element of
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
    **kwargs_fft
        Arbitrary keyword arguments to be passed to the selected fft method

    Attributes
    ----------
    f1 : :obj:`numpy.ndarray`
        Discrete Fourier Transform sample frequencies along ``axes[0]``
    f2 : :obj:`numpy.ndarray`
        Discrete Fourier Transform sample frequencies along ``axes[1]``
    nffts : :obj:`tuple` or :obj:`int`, optional
        Number of samples in Fourier Transform for each axis in ``axes``.
    real : :obj:`bool`
        When ``True``, uses real fast fourier transform.
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
    MPIFFTND: N-dimensional FFT

    Raises
    ------
    ValueError
        - If ``dims`` has less than two elements.
        - If ``axes`` does not have exactly two elements.
        - If ``norm`` is not one of "none", or "1/n".

    Notes
    -----
    The MPIFFT2D operator applies the forward and inverse 2-dimensional Fast Fourier transform to a
    :class:`pylops_mpi.DistributedArray`, which is internally reshaped to the 2-dimensional layout
    defined by ``dims``. The 2-dimensional FFT is then applied across MPI ranks using ``mpi4py_fft``'s
    :class:`mpi4py_fft.mpifft.PFFT` class, with the global array decomposed via a pencil decomposition.
    :class:`mpi4py_fft.pencil.Subcomm` selects the axis of distribution: ``axis=0`` by default,
    shifting to ``axis=1`` if ``axes[-1] == 0`` to avoid a conflict between the transform and
    decomposition axes.

    In the forward pass, :meth:`PFFT.forward` is called with ``normalize=False``, computing:

    .. math::
        D(k_y, k_x) = \mathscr{F} (d) = \iint\limits_{-\infty}^\infty d(y, x) e^{-j2\pi k_yy}
        e^{-j2\pi k_xx} \,\mathrm{d}y \,\mathrm{d}x

    When ``norm="1/n"``, the result is additionally scaled by :math:`1/N_F`.

    In the adjoint pass, :meth:`PFFT.backward` is called with ``normalize=True``, so ``PFFT``
    internally divides by :math:`N_F = N_1 \cdot N_2`, computing:

    .. math::
        d(y,x) = \mathscr{F}^{-1} (D) = \frac{1}{N_F} \iint\limits_{-\infty}^\infty D(k_y, k_x) e^{j2\pi k_yy}
        e^{j2\pi k_xx} \,\mathrm{d}k_y  \,\mathrm{d}k_x

    When ``norm="none"``, the adjoint multiplies by :math:`N_F` to cancel this internal scaling,
    returning a true unscaled adjoint. The result is then flattened back to a 1D
    :class:`pylops_mpi.DistributedArray`. All inter-rank data movement is handled internally by
    ``mpi4py_fft``.
    """
    def __init__(
        self,
        dims: InputDimsLike,
        axes: InputDimsLike = (0, 1),
        sampling: float | Sequence[float] = 1.0,
        norm: str = "none",
        real: bool = False,
        ifftshift_before: bool = False,
        fftshift_after: bool = False,
        dtype: DTypeLike = "complex128",
        base_comm: MPI.Comm = MPI.COMM_WORLD
    ) -> None:
        # checks
        if len(dims) < 2:
            msg = "FFT2D requires at least two input dimensions"
            raise ValueError(msg)
        if len(axes) != 2:
            msg = "FFT2D must be applied along exactly two dimensions"
            raise ValueError(msg)
        super().__init__(dims=dims, axes=axes, sampling=sampling, norm=norm, real=real, dtype=dtype,
                         ifftshift_before=ifftshift_before, fftshift_after=fftshift_after, base_comm=base_comm)
        self.f1, self.f2 = self.fs
        del self.fs

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        return super()._matvec(x)

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        return super()._rmatvec(x)

    def __truediv__(self, y: DistributedArray) -> DistributedArray:
        return super().__truediv__(y)
