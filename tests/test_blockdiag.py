import numpy as np
from numpy.testing import assert_allclose

np.random.seed(42)
import pytest

import pylops
import pylops_mpi
from pylops_mpi import MPILinearOperator

par1 = {'ny': 101, 'nx': 101, 'dtype': np.float64}
par2 = {'ny': 101, 'nx': 101, 'dtype': np.float64}
par3 = {'ny': 301, 'nx': 101, 'dtype': np.float64}
par4 = {'ny': 301, 'nx': 101, 'dtype': np.float64}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_blockdiag(par):
    G1 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G2 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    ops = [pylops.Identity(par['ny'], par['nx']),
           pylops.Zero(par['ny'], par['nx']), pylops.MatrixMult(G1),
           pylops.MatrixMult(G2)]
    mpi_block_diag = pylops_mpi.MPIBlockDiag(ops=ops)
    assert isinstance(mpi_block_diag, MPILinearOperator)
    block_diag = pylops.BlockDiag(ops=ops)
    x = np.random.normal(100, 100, (block_diag.shape[1],))
    y = np.random.normal(100, 100, (block_diag.shape[0],))
    assert mpi_block_diag.shape == block_diag.shape
    assert_allclose(mpi_block_diag * x, block_diag * x, rtol=1e-12)
    assert_allclose(mpi_block_diag.H * y, block_diag.H * y, rtol=1e-12)
