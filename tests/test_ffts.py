"""Test FFT classes
    Designed to run with n processes
    $ mpiexec -n 10 pytest test_ffts.py --with-mpi
"""
import os
import pytest

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    backend = "cupy"
else:
    import numpy as np

    backend = "numpy"

from mpi4py import MPI
from numpy.testing import assert_allclose

from pylops.signalprocessing import FFT2D, FFTND

from pylops_mpi.signalprocessing import MPIFFT2D, MPIFFTND
from pylops_mpi.DistributedArray import DistributedArray

par1 = {
    "dims": (41, 51),
    "axes": (0, 1),
    "real": False,
    "dtype": np.complex128,
    "imag": 1j,
    "norm": "none"
}
par2 = {
    "dims": (50, 50),
    "axes": (0, 1),
    "real": False,
    "dtype": np.complex128,
    "imag": 1j,
    "norm": "1/n"
}
par3 = {
    "dims": (41, 51),
    "axes": (0, 1),
    "real": True,
    "dtype": np.float64,
    "imag": 0,
    "norm": "1/n"
}
par4 = {
    "dims": (50, 50),
    "axes": (0, 1),
    "real": True,
    "dtype": np.float64,
    "imag": 0,
    "norm": "none"
}
par5 = {
    "dims": (41, 51, 50),
    "axes": (0, 1, 2),
    "real": True,
    "dtype": np.float64,
    "imag": 0,
    "norm": "none"
}
par6 = {
    "dims": (41, 51, 50),
    "axes": (0, 2, 1),
    "real": True,
    "dtype": np.float64,
    "imag": 0,
    "norm": "1/n"
}
par7 = {
    "dims": (41, 51, 50),
    "axes": (2, 1, 0),
    "real": False,
    "dtype": np.complex128,
    "imag": 1j,
    "norm": "none"
}
par8 = {
    "dims": (41, 51, 50),
    "axes": (2, 0, 1),
    "real": False,
    "dtype": np.complex128,
    "imag": 1j,
    "norm": "1/n"
}

rank = MPI.COMM_WORLD.Get_rank()


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_FFT2d(par):
    """MPIFFT2D Operator"""
    if backend == "cupy":
        pytest.skip("Skipping cupy backend")
    np.random.seed(10)
    ff2d_mpi = MPIFFT2D(dims=par['dims'], axes=par['axes'], norm=par['norm'], real=par['real'], dtype=par['dtype'])
    x = DistributedArray(global_shape=ff2d_mpi.shape[1], dtype=par['dtype'])
    x[:] = np.random.randn(*(x.local_shape)) + par['imag'] * np.random.randn(*(x.local_shape))
    x_global = x.asarray()
    # Forward
    y_dist = ff2d_mpi @ x
    y = y_dist.asarray()
    # Adjoint
    y_adj_dist = ff2d_mpi.H @ y_dist
    y_adj = y_adj_dist.asarray()
    if rank == 0:
        fft2d = FFT2D(dims=par['dims'], axes=par['axes'], norm=par['norm'], real=par['real'], dtype=par['dtype'])
        assert ff2d_mpi.shape == fft2d.shape
        y_np = fft2d @ x_global
        y_adj_np = fft2d.H @ y_np
        assert_allclose(y, y_np, rtol=1e-5, atol=1e-8)
        assert_allclose(y_adj, y_adj_np, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6), (par7), (par8)])
def test_FFTND(par):
    """MPIFFTND Operator"""
    if backend == "cupy":
        pytest.skip("Skipping cupy backend")
    np.random.seed(10)
    ffnd_mpi = MPIFFTND(dims=par['dims'], axes=par['axes'], norm=par['norm'], real=par['real'], dtype=par['dtype'])
    x = DistributedArray(global_shape=ffnd_mpi.shape[1], dtype=par['dtype'])
    x[:] = np.random.randn(*(x.local_shape)) + par['imag'] * np.random.randn(*(x.local_shape))
    x_global = x.asarray()
    # Forward
    y_dist = ffnd_mpi @ x
    y = y_dist.asarray()
    # Adjoint
    y_adj_dist = ffnd_mpi.H @ y_dist
    y_adj = y_adj_dist.asarray()
    # Div
    y_div_dist = ffnd_mpi / y_dist
    y_div = y_div_dist.asarray()
    if rank == 0:
        fftnd = FFTND(dims=par['dims'], axes=par['axes'], norm=par['norm'], real=par['real'], dtype=par['dtype'])
        assert ffnd_mpi.shape == fftnd.shape
        y_np = fftnd @ x_global
        y_adj_np = fftnd.H @ y_np
        y_div_np = fftnd / y_np
        assert_allclose(y, y_np, rtol=1e-5, atol=1e-8)
        assert_allclose(y_adj, y_adj_np, rtol=1e-5, atol=1e-8)
        assert_allclose(y_div, y_div_np, rtol=1e-5, atol=1e-8)
