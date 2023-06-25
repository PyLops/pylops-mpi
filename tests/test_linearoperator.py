import numpy as np
from numpy.testing import assert_allclose

np.random.seed(42)
import pytest

import pylops
from pylops_mpi import asmpilinearoperator, DistributedArray, MPILinearOperator

par1 = {"ny": 11, "nx": 11, "dtype": np.float64}  # square real
par2 = {"ny": 21, "nx": 11, "dtype": np.float64}  # overdetermined real
par1j = {"ny": 11, "nx": 11, "dtype": np.float64}  # square imag
par2j = {"ny": 21, "nx": 11, "dtype": np.float64}  # overdetermined imag


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_linearop(par):
    # for kind = "all"
    diag = DistributedArray.to_dist(x=np.arange(par['nx'] + par['ny']))
    Dop = pylops.Diagonal(diag.local_array, dtype=par["dtype"])
    Dop_MPI_1 = asmpilinearoperator(Op=Dop, kind="all")

    # for kind = "master"
    diag = np.arange(par['nx'] + par['ny'])
    Dop = pylops.Diagonal(diag, dtype=par['dtype'])
    Dop_MPI_2 = asmpilinearoperator(Op=Dop, kind="master")

    assert isinstance(Dop_MPI_1, MPILinearOperator)
    assert isinstance(Dop_MPI_2, MPILinearOperator)

    assert isinstance(Dop_MPI_1.T, MPILinearOperator)
    assert isinstance(Dop_MPI_2.T, MPILinearOperator)

    assert isinstance(Dop_MPI_1.H, MPILinearOperator)
    assert isinstance(Dop_MPI_2.H, MPILinearOperator)

    assert isinstance(Dop_MPI_1.conj(), MPILinearOperator)
    assert isinstance(Dop_MPI_2.conj(), MPILinearOperator)

    assert isinstance(Dop_MPI_1 + Dop_MPI_2, MPILinearOperator)

    assert isinstance(Dop_MPI_2 * Dop_MPI_1, MPILinearOperator)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_scaledop(par):
    diag = np.arange(par['nx'] + par['ny'])
    Dop = pylops.Diagonal(diag=diag)
    Sop = Dop * 5

    diag = DistributedArray.to_dist(x=diag)
    Dop = asmpilinearoperator(pylops.Diagonal(diag=diag.local_array))
    Sop_MPI = Dop * 5

    x = np.arange(par['nx'] + par['ny'])
    assert_allclose(Sop_MPI * x, Sop * x, rtol=1e-14)
    assert_allclose(Sop_MPI.H * x, Sop.H * x, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_conj(par):
    diag = np.arange(par['nx'] + par['ny'])
    Dop = pylops.Diagonal(diag=diag)
    Cop = Dop.conj()

    diag = DistributedArray.to_dist(x=diag)
    Dop = asmpilinearoperator(pylops.Diagonal(diag=diag.local_array))
    Cop_MPI = Dop.conj()

    x = np.arange(par['nx'] + par['ny'])
    assert_allclose(Cop_MPI * x, Cop * x, rtol=1e-14)
    assert_allclose(Cop_MPI.H * x, Cop.H * x, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_power(par):
    diag = np.arange(par['nx'] + par['ny'])
    Dop = pylops.Diagonal(diag=diag)
    Pop = Dop ** 2

    diag = DistributedArray.to_dist(x=diag)
    Dop = asmpilinearoperator(pylops.Diagonal(diag=diag.local_array))
    Pop_MPI = Dop ** 2

    x = np.arange(par['nx'] + par['ny'])
    assert_allclose(Pop_MPI * x, Pop * x, rtol=1e-14)
    assert_allclose(Pop_MPI.H * x, Pop.H * x, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_sum_prod(par):
    diag = np.arange(par['nx'] + par['ny'])
    Dop = pylops.Diagonal(diag=diag)
    DDop = pylops.Diagonal(diag=2 * diag)

    Sop = Dop + DDop
    Mop = Dop * DDop

    diag = DistributedArray.to_dist(x=diag)
    Dop = asmpilinearoperator(pylops.Diagonal(diag=diag.local_array))
    DDop = asmpilinearoperator(pylops.Diagonal(diag=2 * diag.local_array))

    Sop_MPI = Dop + DDop
    Mop_MPI = Dop * DDop

    x = np.arange(par['nx'] + par['ny'])
    assert_allclose(Sop_MPI * x, Sop * x, rtol=1e-14)
    assert_allclose(Sop_MPI.H * x, Sop.H * x, rtol=1e-14)
    assert_allclose(Mop_MPI * x, Mop * x, rtol=1e-14)
    assert_allclose(Mop_MPI.H * x, Mop.H * x, rtol=1e-14)
