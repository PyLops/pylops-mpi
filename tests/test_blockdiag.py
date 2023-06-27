import numpy as np
from numpy.testing import assert_allclose

np.random.seed(42)
import pytest

import pylops
import pylops_mpi

par1 = {'ny': 101, 'nx': 101, 'dtype': np.float64}
par1j = {'ny': 101, 'nx': 101, 'dtype': np.complex128}
par2 = {'ny': 301, 'nx': 101, 'dtype': np.float64}
par2j = {'ny': 301, 'nx': 101, 'dtype': np.complex128}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_blockdiag(par):
    G1 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G2 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    ops = [pylops.Identity(par['ny'], par['nx']),
           pylops.Zero(par['ny'], par['nx']), pylops.MatrixMult(G1),
           pylops.MatrixMult(G2)]
    BDiag_MPI = pylops_mpi.MPIBlockDiag(ops=ops)
    assert isinstance(BDiag_MPI, pylops.LinearOperator)
    BDiag = pylops.BlockDiag(ops=ops)
    x = np.random.normal(100, 100, (BDiag.shape[1],))
    y = np.random.normal(100, 100, (BDiag.shape[0],))
    assert BDiag_MPI.shape == BDiag.shape
    assert_allclose(BDiag_MPI * x, BDiag * x, rtol=1e-14)
    assert_allclose(BDiag_MPI.H * y, BDiag.H * y, rtol=1e-14)
