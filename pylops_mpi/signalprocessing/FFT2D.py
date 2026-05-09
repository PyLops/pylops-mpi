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
    fft : :obj:`mpi4py_fft.PFFT`
        Parallel FFT operator object handling the distributed transform across
        MPI processes. Configured with the base communicator, dimension
        decomposition, transform axes, and dtype.

    See Also
    --------
    MPIFFTND: N-dimensional FFT

    Raises
    ------
    ValueError
        - If ``norm`` is not one of "none", or "1/n".

    Notes
    -----
    The FFT2D operator (using ``norm="none"``) applies the two-dimensional forward
    Fourier transform to a signal :math:`d(y, x)` in forward mode:

    .. math::
        D(k_y, k_x) = \mathscr{F} (d) = \iint\limits_{-\infty}^\infty d(y, x) e^{-j2\pi k_yy}
        e^{-j2\pi k_xx} \,\mathrm{d}y \,\mathrm{d}x

    Similarly, the two-dimensional inverse Fourier transform is applied to
    the Fourier spectrum :math:`D(k_y, k_x)` in adjoint mode:

    .. math::
        d(y,x) = \mathscr{F}^{-1} (D) = \frac{1}{N_F} \iint\limits_{-\infty}^\infty D(k_y, k_x) e^{j2\pi k_yy}
        e^{j2\pi k_xx} \,\mathrm{d}k_y  \,\mathrm{d}k_x

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
        axes: InputDimsLike = (0, 1),
        sampling: float | Sequence[float] = 1.0,
        norm: str = "none",
        real: bool = False,
        dtype: DTypeLike = "complex128",
        base_comm: MPI.Comm = MPI.COMM_WORLD,
        **kwargs_fft,
    ) -> None:
        # checks
        if len(dims) < 2:
            msg = "FFT2D requires at least two input dimensions"
            raise ValueError(msg)
        if len(axes) != 2:
            msg = "FFT2D must be applied along exactly two dimensions"
            raise ValueError(msg)
        super().__init__(dims=dims, axes=axes, sampling=sampling, norm=norm, real=real,
                         dtype=dtype, base_comm=base_comm, **kwargs_fft)
        self.f1, self.f2 = self.fs
        del self.fs

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        return super()._matvec(x)

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        return super()._rmatvec(x)

    def __truediv__(self, y: DistributedArray) -> DistributedArray:
        return super().__truediv__(y)
