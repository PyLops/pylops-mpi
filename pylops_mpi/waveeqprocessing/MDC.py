import logging
import numpy as np

from mpi4py import MPI
from pylops import Identity
from pylops.signalprocessing import FFT, Fredholm1

from pylops_mpi import MPILinearOperator
from pylops_mpi.signalprocessing.Fredholm1 import MPIFredholm1


def _MDC(G, nt, nv, nfmax, dt=1., dr=1., twosided=True,
         saveGt=True, conj=False, prescaled=False,
         base_comm=MPI.COMM_WORLD,
         _Identity=Identity, _FFT=FFT,
         _Fredholm1=Fredholm1, args_Identity={},
         args_FFT={}, args_Identity1={},
         args_FFT1={}, args_Fredholm1={}):
    r"""Multi-dimensional convolution.

    Used to be able to provide operators from different libraries to
    MDC. It operates in the same way as public method
    (MPIMDC) but has additional input parameters allowing
    passing a different operator and additional arguments to be passed to such
    operator.

    """
    if twosided and nt % 2 == 0:
        raise ValueError('nt must be odd number')

    # find out dtype of G
    dtype = G[0, 0, 0].dtype
    rdtype = np.real(np.ones(1, dtype=dtype)).dtype

    # create Fredholm operator
    if prescaled:
        Frop = _Fredholm1(G, nv, saveGt=saveGt,
                          base_comm=base_comm,
                          dtype=dtype, **args_Fredholm1)
    else:
        Frop = _Fredholm1(dr * dt * np.sqrt(nt) * G, nv,
                          saveGt=saveGt, base_comm=base_comm,
                          dtype=dtype, **args_Fredholm1)
    if conj:
        Frop = Frop.conj()

    # create FFT operators
    _, ns, nr = G.shape
    # ensure that nfmax is not bigger than allowed
    nfft = int(np.ceil((nt + 1) / 2))
    if nfmax > nfft:
        nfmax = nfft
        logging.warning('nfmax set equal to ceil[(nt+1)/2=%d]' % nfmax)

    Fop = MPILinearOperator(_FFT(dims=(nt, nr, nv), axis=0, real=True,
                                 ifftshift_before=twosided, dtype=rdtype, **args_FFT))
    F1op = MPILinearOperator(_FFT(dims=(nt, ns, nv), axis=0, real=True,
                                  ifftshift_before=False, dtype=rdtype, **args_FFT1))

    # create Identity operator to extract only relevant frequencies
    Iop = MPILinearOperator(_Identity(N=nfmax * nr * nv, M=nfft * nr * nv,
                                      inplace=True, dtype=dtype, **args_Identity))
    I1op = MPILinearOperator(_Identity(N=nfmax * ns * nv, M=nfft * ns * nv,
                                       inplace=True, dtype=dtype, **args_Identity1))
    F1opH = F1op.H
    I1opH = I1op.H

    # create MDC operator
    MDCop = F1opH * I1opH * Frop * Iop * Fop

    # force dtype to be real (as FFT operators assume real inputs and outputs)
    MDCop.dtype = rdtype

    return MDCop


def MPIMDC(G, nt, nv, nfreq, dt=1., dr=1., twosided=True,
           fftengine='numpy',
           saveGt=True, conj=False,
           usematmul=False, prescaled=False,
           base_comm: MPI.Comm = MPI.COMM_WORLD):
    r"""Multi-dimensional convolution.

    Apply multi-dimensional convolution between two datasets in a distributed
    fashion, with ``G`` distributed over ranks across the frequency axis.
    Model and data are broadcasted and should be provided after flattening
    2- or 3-dimensional arrays of size
    :math:`[n_t \times n_r (\times n_{vs})]` and
    :math:`[n_t \times n_s (\times n_{vs})]` (or :math:`2*n_t-1` for
    ``twosided=True``), respectively.

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel in frequency domain of size
        :math:`[n_{f,rank} \times n_s \times n_r]`
    nt : :obj:`int`
        Number of samples along time axis for model and data (note that this
        must be equal to ``2*n_t-1`` when working with ``twosided=True``.
    nv : :obj:`int`
        Number of samples along virtual source axis
    dt : :obj:`float`, optional
        Sampling of time integration axis
    dr : :obj:`float`, optional
        Sampling of receiver integration axis
    twosided : :obj:`bool`, optional
        MDC operator has both negative and positive time (``True``) or
        only positive (``False``)
    fftengine : :obj:`str`, optional
        Engine used for fft computation (``numpy`` or ``fftw``)
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G^H`` to speed up the computation of adjoint of
        :class:`pylops.signalprocessing.Fredholm1` (``True``) or create
        ``G^H`` on-the-fly (``False``) Note that ``saveGt=True`` will be
        faster but double the amount of required memory
    conj : :obj:`str`, optional
        Perform Fredholm integral computation with complex conjugate of ``G``
    usematmul : :obj:`bool`, optional
        Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
        (``False``) in :py:class:`pylops.signalprocessing.Fredholm1` operator.
        Refer to Fredholm1 documentation for details.
    prescaled : :obj:`bool`, optional
        Apply scaling to kernel (``False``) or not (``False``) when performing
        spatial and temporal summations. In case ``prescaled=True``, the
        kernel is assumed to have been pre-scaled when passed to the MDC
        routine.
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        MPI Base Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.

    Raises
    ------
    ValueError
        If ``nt`` is even and ``twosided=True``

    See Also
    --------
    MDD : Multi-dimensional deconvolution

    Notes
    -----
    The so-called multi-dimensional convolution (MDC) is a chained
    operator [1]_. It is composed of a forward Fourier transform,
    a multi-dimensional integration, and an inverse Fourier transform:

    .. math::
        y(t, s, v) = \mathscr{F}^{-1} \Big( \int_S G(f, s, r)
        \mathscr{F}(x(t, r, v)) dr \Big)

    which is discretized as follows:

    .. math::
        y(t, s, v) = \mathscr{F}^{-1} \Big( \sum_{i_r=0}^{n_r}
        (\sqrt{n_t} * d_t * d_r) G(f, s, i_r) \mathscr{F}(x(t, i_r, v)) \Big)

    where :math:`(\sqrt{n_t} * d_t * d_r)` is not applied if ``prescaled=True``.

    This operation can be discretized and performed by means of a
    linear operator

    .. math::
        \mathbf{D}= \mathbf{F}^H  \mathbf{G} \mathbf{F}

    where :math:`\mathbf{F}` is the Fourier transform applied along
    the time axis and :math:`\mathbf{G}` is the multi-dimensional
    convolution kernel.

    .. [1] Wapenaar, K., van der Neut, J., Ruigrok, E., Draganov, D., Hunziker,
       J., Slob, E., Thorbecke, J., and Snieder, R., "Seismic interferometry
       by crosscorrelation and by multi-dimensional deconvolution: a
       systematic comparison", Geophysical Journal International, vol. 185,
       pp. 1335-1364. 2011.

    """
    return _MDC(G, nt, nv, nfreq, dt=dt, dr=dr, twosided=twosided,
                saveGt=saveGt, conj=conj, prescaled=prescaled,
                base_comm=base_comm,
                _Fredholm1=MPIFredholm1,
                args_FFT={"engine": fftengine},
                args_FFT1={"engine": fftengine},
                args_Fredholm1={'usematmul': usematmul})
