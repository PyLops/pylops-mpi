import numpy as np
from mpi4py import MPI
from numpy.testing import assert_allclose
import pytest

import pylops
import pylops_mpi

np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

par1 = {
    "nz": 100,
    "dz": 1.0,
    "edge": False,
    "dtype": np.float128,
    "partition": pylops_mpi.Partition.SCATTER
}

par1b = {
    "nz": 100,
    "dz": 1.0,
    "edge": False,
    "dtype": np.float128,
    "partition": pylops_mpi.Partition.BROADCAST
}

par1j = {
    "nz": 100,
    "dz": 1.0,
    "edge": False,
    "dtype": np.complex256,
    "partition": pylops_mpi.Partition.SCATTER
}

par1e = {
    "nz": 100,
    "dz": 1.0,
    "edge": True,
    "dtype": np.float128,
    "partition": pylops_mpi.Partition.SCATTER
}

par2 = {
    "nz": (101, 100),
    "dz": 1.0,
    "edge": False,
    "dtype": np.float128,
    "partition": pylops_mpi.Partition.SCATTER
}

par2b = {
    "nz": (101, 100),
    "dz": 1.0,
    "edge": False,
    "dtype": np.float128,
    "partition": pylops_mpi.Partition.BROADCAST
}

par2j = {
    "nz": (101, 100),
    "dz": 1.0,
    "edge": False,
    "dtype": np.complex256,
    "partition": pylops_mpi.Partition.SCATTER
}

par2e = {
    "nz": (101, 100),
    "dz": 1.0,
    "edge": True,
    "dtype": np.float128,
    "partition": pylops_mpi.Partition.SCATTER
}

par3 = {
    "nz": (101, 50, 100),
    "dz": 0.4,
    "edge": False,
    "dtype": np.float128,
    "partition": pylops_mpi.Partition.SCATTER
}

par3b = {
    "nz": (101, 50, 100),
    "dz": 0.4,
    "edge": False,
    "dtype": np.float128,
    "partition": pylops_mpi.Partition.BROADCAST
}

par3j = {
    "nz": (101, 50, 100),
    "dz": 0.4,
    "edge": True,
    "dtype": np.complex256,
    "partition": pylops_mpi.Partition.SCATTER
}

par3e = {
    "nz": (101, 50, 100),
    "dz": 0.4,
    "edge": True,
    "dtype": np.float128,
    "partition": pylops_mpi.Partition.SCATTER
}

par4 = {
    "nz": (101, 101, 101),
    "dz": 0.4,
    "edge": False,
    "dtype": np.float128,
    "partition": pylops_mpi.Partition.SCATTER
}

par4b = {
    "nz": (101, 101, 101),
    "dz": 0.4,
    "edge": False,
    "dtype": np.float128,
    "partition": pylops_mpi.Partition.BROADCAST
}

par4j = {
    "nz": (101, 101, 101),
    "dz": 0.4,
    "edge": True,
    "dtype": np.complex256,
    "partition": pylops_mpi.Partition.SCATTER
}

par4e = {
    "nz": (101, 101, 101),
    "dz": 0.4,
    "edge": True,
    "dtype": np.float128,
    "partition": pylops_mpi.Partition.SCATTER
}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1b), (par1j), (par1e), (par2), (par2b),
                                 (par2j), (par2e), (par3), (par3b), (par3j), (par3e),
                                 (par4), (par4b), (par4j), (par4e)])
def test_first_derivative_forward(par):
    Fop_MPI = pylops_mpi.MPIFirstDerivative(dims=par['nz'], sampling=par['dz'],
                                            kind="forward", edge=par['edge'],
                                            dtype=par['dtype'])
    x = pylops_mpi.DistributedArray(global_shape=np.prod(par['nz']), dtype=par['dtype'],
                                    partition=par['partition'])
    x[:] = np.random.normal(rank, 10, x.local_shape)
    x_global = x.asarray()
    # Forward
    y_dist = Fop_MPI @ x
    y = y_dist.asarray()
    # Adjoint
    y_adj_dist = Fop_MPI.H @ x
    y_adj = y_adj_dist.asarray()

    if rank == 0:
        Fop = pylops.FirstDerivative(dims=par['nz'], axis=0,
                                     sampling=par['dz'],
                                     kind="forward", edge=par['edge'],
                                     dtype=par['dtype'])
        assert Fop_MPI.shape == Fop.shape
        y_np = Fop @ x_global
        y_adj_np = Fop.H @ x_global
        assert_allclose(y, y_np, rtol=1e-14)
        assert_allclose(y_adj, y_adj_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1b), (par1j), (par1e), (par2), (par2b),
                                 (par2j), (par2e), (par3), (par3b), (par3j), (par3e),
                                 (par4), (par4b), (par4j), (par4e)])
def test_first_derivative_backward(par):
    Fop_MPI = pylops_mpi.MPIFirstDerivative(dims=par['nz'], sampling=par['dz'],
                                            kind="backward", edge=par['edge'],
                                            dtype=par['dtype'])
    x = pylops_mpi.DistributedArray(global_shape=np.prod(par['nz']), dtype=par['dtype'],
                                    partition=par['partition'])
    x[:] = np.random.normal(rank, 10, x.local_shape)
    x_global = x.asarray()
    # Forward
    y_dist = Fop_MPI @ x
    y = y_dist.asarray()
    # Adjoint
    y_adj_dist = Fop_MPI.H @ x
    y_adj = y_adj_dist.asarray()

    if rank == 0:
        Fop = pylops.FirstDerivative(dims=par['nz'], axis=0,
                                     sampling=par['dz'],
                                     kind="backward", edge=par['edge'],
                                     dtype=par['dtype'])
        assert Fop_MPI.shape == Fop.shape
        y_np = Fop @ x_global
        y_adj_np = Fop.H @ x_global
        assert_allclose(y, y_np, rtol=1e-14)
        assert_allclose(y_adj, y_adj_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1b), (par1j), (par1e), (par2), (par2b),
                                 (par2j), (par2e), (par3), (par3b), (par3j), (par3e),
                                 (par4), (par4b), (par4j), (par4e)])
def test_first_derivative_centered(par):
    for order in [3, 5]:
        Fop_MPI = pylops_mpi.MPIFirstDerivative(dims=par['nz'], sampling=par['dz'],
                                                kind="centered", edge=par['edge'],
                                                order=order, dtype=par['dtype'])
        x = pylops_mpi.DistributedArray(global_shape=np.prod(par['nz']), dtype=par['dtype'],
                                        partition=par['partition'])
        x[:] = np.random.normal(rank, 10, x.local_shape)
        x_global = x.asarray()
        # Forward
        y_dist = Fop_MPI @ x
        y = y_dist.asarray()
        # Adjoint
        y_adj_dist = Fop_MPI.H @ x
        y_adj = y_adj_dist.asarray()
        if rank == 0:
            Fop = pylops.FirstDerivative(dims=par['nz'], axis=0,
                                         sampling=par['dz'],
                                         kind="centered", edge=par['edge'],
                                         order=order, dtype=par['dtype'])
            assert Fop_MPI.shape == Fop.shape
            y_np = Fop @ x_global
            y_adj_np = Fop.H @ x_global
            assert_allclose(y, y_np, rtol=1e-14)
            assert_allclose(y_adj, y_adj_np, rtol=1e-14)
