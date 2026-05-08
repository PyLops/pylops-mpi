import warnings
from typing import Sequence

from mpi4py import MPI
import numpy as np

from pylops.signalprocessing._baseffts import _FFTNorms
from pylops.utils import InputDimsLike, DTypeLike, get_normalize_axis_index, get_real_dtype, get_complex_dtype
from pylops.utils._internal import _value_or_sized_to_array, _raise_on_wrong_dtype, _value_or_sized_to_tuple

from pylops_mpi.DistributedArray import DistributedArray
from pylops_mpi.LinearOperator import MPILinearOperator


class _MPIBaseFFTND(MPILinearOperator):
    """Base class for N-dimensional fast Fourier Transform"""

    def __init__(
        self,
        dims: int | InputDimsLike,
        axes: int | InputDimsLike | None = None,
        nffts: int | InputDimsLike | None = None,
        sampling: float | Sequence[float] = 1.0,
        norm: str = "none",
        real: bool = False,
        dtype: DTypeLike = "complex128",
        base_comm: MPI.Comm = MPI.COMM_WORLD
    ):
        dims = _value_or_sized_to_array(dims)
        _raise_on_wrong_dtype(dims, np.integer, "dims")
        self.dims = tuple(dims)
        self.ndim = len(dims)

        axes = _value_or_sized_to_array(axes)
        _raise_on_wrong_dtype(axes, np.integer, "axes")
        self.axes = np.array([get_normalize_axis_index()(d, self.ndim) for d in axes])
        self.naxes = len(self.axes)
        if self.naxes != len(np.unique(self.axes)):
            warnings.warn(
                "At least one direction is repeated. This may cause unexpected results.",
                stacklevel=2,
            )

        nffts = _value_or_sized_to_array(nffts, repeat=self.naxes)
        if len(nffts[np.equal(nffts, None)]) > 0:  # Found None(s) in nffts
            nffts[np.equal(nffts, None)] = np.array(
                [dims[d] for d, n in zip(axes, nffts, strict=True) if n is None]
            )
            nffts = nffts.astype(np.array(dims).dtype)
        _raise_on_wrong_dtype(nffts, np.integer, "nffts")
        self.nffts = _value_or_sized_to_tuple(
            nffts
        )  # tuple is strictly needed for cupy

        sampling = _value_or_sized_to_array(sampling, repeat=self.naxes)
        if np.issubdtype(sampling.dtype, np.integer):  # Promote to float64 if integer
            sampling = sampling.astype(np.float64)
        self.sampling = sampling
        _raise_on_wrong_dtype(self.sampling, np.floating, "sampling")

        if (
            self.naxes != len(self.nffts)
            or self.naxes != len(self.sampling)
        ):
            msg = (
                "`axes`, `nffts`, `sampling` must the have same number of elements. Received "
                f"{self.naxes}, {len(self.nffts)}, {len(self.sampling)}, "
                "respectively."
            )
            raise ValueError(msg)

        # Check if the user provided nfft smaller than n. See _BaseFFT for
        # details
        nfftshort = [
            nfft < dims[direction]
            for direction, nfft in zip(self.axes, self.nffts, strict=True)
        ]
        self.doifftpad = any(nfftshort)
        if self.doifftpad:
            self.ifftpad = [(0, 0)] * self.ndim
            for idir, (direction, nfshort) in enumerate(
                zip(self.axes, nfftshort, strict=True)
            ):
                if nfshort:
                    self.ifftpad[direction] = (
                        0,
                        dims[direction] - self.nffts[idir],
                    )
            warnings.warn(
                f"nffts in directions {np.where(nfftshort)[0]} have been selected to be smaller than the size of the original signal. "
                "This is rarely intended behavior as the original signal will be truncated prior to applying fft, "
                f"if this is the required behaviour ignore this message.",
                stacklevel=2,
            )

        if norm == "none":
            self.norm = _FFTNorms.NONE
        elif norm.lower() == "1/n":
            self.norm = _FFTNorms.ONE_OVER_N
        elif norm == "backward":
            msg = 'To use no scaling on the forward transform, use "none". Note that in this case, the adjoint transform will *not* have a 1/n scaling.'
            raise ValueError(msg)
        elif norm == "forward":
            msg = 'To use 1/n scaling on the forward transform, use "1/n". Note that in this case, the adjoint transform will *also* have a 1/n scaling.'
            raise ValueError(msg)
        else:
            msg = f"`norm`={norm} is not one of 'none' or '1/n'"
            raise ValueError(msg)

        self.real = real

        fs = [np.fft.fftfreq(n, d=s) for n, s in zip(self.nffts, self.sampling, strict=True)]
        if self.real:
            fs[-1] = np.fft.rfftfreq(self.nffts[-1], d=self.sampling[-1])
        self.fs = tuple(fs)
        dimsd = np.array(dims)
        dimsd[self.axes] = self.nffts
        if self.real:
            dimsd[self.axes[-1]] = self.nffts[-1] // 2 + 1
        self.dimsd = dimsd
        # Find types to enforce to forward and adjoint outputs. This is
        # required as np.fft.fft always returns complex128 even if input is
        # float32 or less. Moreover, when choosing real=True, the type of the
        # adjoint output is forced to be real even if the provided dtype
        # is complex.
        self.rdtype = get_real_dtype(dtype) if self.real else np.dtype(dtype)
        self.cdtype = get_complex_dtype(dtype)
        self.clinear = False if self.real or np.issubdtype(dtype, np.floating) else True
        super().__init__(dtype=self.cdtype, shape=(int(np.prod(dimsd)), int(np.prod(dims))), base_comm=base_comm)

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        msg = "_BaseFFT does not provide _matvec. It must be implemented separately."
        raise NotImplementedError(msg)

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        msg = "_BaseFFT does not provide _rmatvec. It must be implemented separately."
        raise NotImplementedError(msg)
