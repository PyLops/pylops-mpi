"""Test the derivative classes
    Designed to run with n GPUs (with 1 MPI process per GPU)
    $ mpiexec -n 10 pytest test_derivative_nccl.py --with-mpi

This file employs the same test sets as test_derivative under NCCL environment
"""

import numpy as np
import cupy as cp
from mpi4py import MPI
from numpy.testing import assert_allclose
import pytest

import pylops
import pylops_mpi
from pylops_mpi.utils.dottest import dottest
from pylops_mpi.utils._nccl import initialize_nccl_comm

nccl_comm = initialize_nccl_comm()

np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

par1 = {
    "nz": 600,
    "dz": 1.0,
    "edge": False,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.SCATTER,
}

par1b = {
    "nz": 600,
    "dz": 1.0,
    "edge": False,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.BROADCAST,
}

par1j = {
    "nz": 600,
    "dz": 1.0,
    "edge": False,
    "dtype": np.complex128,
    "partition": pylops_mpi.Partition.SCATTER
}

par1e = {
    "nz": 600,
    "dz": 1.0,
    "edge": True,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.SCATTER,
}

par2 = {
    "nz": (100, 151),
    "dz": 1.0,
    "edge": False,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.SCATTER,
}

par2b = {
    "nz": (100, 151),
    "dz": 1.0,
    "edge": False,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.BROADCAST,
}

par2j = {
    "nz": (100, 151),
    "dz": 1.0,
    "edge": False,
    "dtype": np.complex128,
    "partition": pylops_mpi.Partition.SCATTER
}

par2e = {
    "nz": (100, 151),
    "dz": 1.0,
    "edge": True,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.SCATTER,
}

par3 = {
    "nz": (101, 51, 100),
    "dz": 0.4,
    "edge": False,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.SCATTER,
}

par3b = {
    "nz": (101, 51, 100),
    "dz": 0.4,
    "edge": False,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.BROADCAST,
}

par3j = {
    "nz": (101, 51, 100),
    "dz": 0.4,
    "edge": True,
    "dtype": np.complex128,
    "partition": pylops_mpi.Partition.SCATTER
}

par3e = {
    "nz": (101, 51, 100),
    "dz": 0.4,
    "edge": True,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.SCATTER,
}

par4 = {
    "nz": (79, 101, 50),
    "dz": 0.4,
    "edge": False,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.SCATTER,
}

par4b = {
    "nz": (79, 101, 50),
    "dz": 0.4,
    "edge": False,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.BROADCAST,
}

par4j = {
    "nz": (79, 101, 50),
    "dz": 0.4,
    "edge": True,
    "dtype": np.complex128,
    "partition": pylops_mpi.Partition.SCATTER
}

par4e = {
    "nz": (79, 101, 50),
    "dz": 0.4,
    "edge": True,
    "dtype": np.float64,
    "partition": pylops_mpi.Partition.SCATTER,
}

par5 = {
    "n": (120, 101, 60),
    "axes": (0, 1, 2),
    "weights": (0.7, 0.7, 0.7),
    "sampling": (1, 1, 1),
    "edge": False,
    "dtype": np.float64,
}

par5e = {
    "n": (120, 101, 60),
    "axes": (-1, -2, -3),
    "weights": (0.7, 0.7, 0.7),
    "sampling": (1, 1, 1),
    "edge": True,
    "dtype": np.float64,
}

par6 = {
    "n": (79, 60, 101),
    "axes": (0, 1, 2),
    "weights": (1, 1, 1),
    "sampling": (0.4, 0.4, 0.4),
    "edge": False,
    "dtype": np.float64,
}

par6e = {
    "n": (79, 60, 101),
    "axes": (-1, -2, -3),
    "weights": (1, 1, 1),
    "sampling": (0.4, 0.4, 0.4),
    "edge": True,
    "dtype": np.float64,
}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1b), (par1j), (par1e), (par2), (par2b),
                                 (par2j), (par2e), (par3), (par3b), (par3j), (par3e),
                                 (par4), (par4b), (par4j), (par4e)])
def test_first_derivative_forward_nccl(par):
    """MPIFirstDerivative operator (forward stencil)"""
    Fop_MPI = pylops_mpi.MPIFirstDerivative(
        dims=par["nz"],
        sampling=par["dz"],
        kind="forward",
        edge=par["edge"],
        dtype=par["dtype"],
    )
    x = pylops_mpi.DistributedArray(
        global_shape=np.prod(par["nz"]),
        base_comm_nccl=nccl_comm,
        dtype=par["dtype"],
        partition=par["partition"],
        engine="cupy",
    )
    x[:] = cp.random.normal(rank, 10, x.local_shape)
    x_global = x.asarray()
    # Forward
    y_dist = Fop_MPI @ x
    y = y_dist.asarray()
    # Adjoint
    y_adj_dist = Fop_MPI.H @ x
    y_adj = y_adj_dist.asarray()

    # Dot test
    dottest(Fop_MPI, x, y_dist, np.prod(par["nz"]), np.prod(par["nz"]))

    if rank == 0:
        Fop = pylops.FirstDerivative(
            dims=par["nz"],
            axis=0,
            sampling=par["dz"],
            kind="forward",
            edge=par["edge"],
            dtype=par["dtype"],
        )
        assert Fop_MPI.shape == Fop.shape
        y_np = Fop @ x_global.get()
        y_adj_np = Fop.H @ x_global.get()
        assert_allclose(y.get(), y_np, rtol=1e-14)
        assert_allclose(y_adj.get(), y_adj_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1b), (par1j), (par1e), (par2), (par2b),
                                 (par2j), (par2e), (par3), (par3b), (par3j), (par3e),
                                 (par4), (par4b), (par4j), (par4e)])
def test_first_derivative_backward_nccl(par):
    """MPIFirstDerivative operator (backward stencil)"""
    Fop_MPI = pylops_mpi.MPIFirstDerivative(
        dims=par["nz"],
        sampling=par["dz"],
        kind="backward",
        edge=par["edge"],
        dtype=par["dtype"],
    )
    x = pylops_mpi.DistributedArray(
        global_shape=np.prod(par["nz"]),
        base_comm_nccl=nccl_comm,
        dtype=par["dtype"],
        partition=par["partition"],
        engine="cupy",
    )
    x[:] = cp.random.normal(rank, 10, x.local_shape)
    x_global = x.asarray()
    # Forward
    y_dist = Fop_MPI @ x
    y = y_dist.asarray()
    # Adjoint
    y_adj_dist = Fop_MPI.H @ x
    y_adj = y_adj_dist.asarray()
    # Dot test
    dottest(Fop_MPI, x, y_dist, np.prod(par["nz"]), np.prod(par["nz"]))

    if rank == 0:
        Fop = pylops.FirstDerivative(
            dims=par["nz"],
            axis=0,
            sampling=par["dz"],
            kind="backward",
            edge=par["edge"],
            dtype=par["dtype"],
        )
        assert Fop_MPI.shape == Fop.shape
        y_np = Fop @ x_global.get()
        y_adj_np = Fop.H @ x_global.get()
        assert_allclose(y.get(), y_np, rtol=1e-14)
        assert_allclose(y_adj.get(), y_adj_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1b), (par1j), (par1e), (par2), (par2b),
                                 (par2j), (par2e), (par3), (par3b), (par3j), (par3e),
                                 (par4), (par4b), (par4j), (par4e)])
def test_first_derivative_centered_nccl(par):
    """MPIFirstDerivative operator (centered stencil)"""
    for order in [3, 5]:
        Fop_MPI = pylops_mpi.MPIFirstDerivative(
            dims=par["nz"],
            sampling=par["dz"],
            kind="centered",
            edge=par["edge"],
            order=order,
            dtype=par["dtype"],
        )
        x = pylops_mpi.DistributedArray(
            global_shape=np.prod(par["nz"]),
            base_comm_nccl=nccl_comm,
            dtype=par["dtype"],
            partition=par["partition"],
            engine="cupy",
        )
        x[:] = cp.random.normal(rank, 10, x.local_shape)
        x_global = x.asarray()
        # Forward
        y_dist = Fop_MPI @ x
        y = y_dist.asarray()
        # Adjoint
        y_adj_dist = Fop_MPI.H @ x
        y_adj = y_adj_dist.asarray()
        # Dot test
        dottest(Fop_MPI, x, y_dist, np.prod(par["nz"]), np.prod(par["nz"]))

        if rank == 0:
            Fop = pylops.FirstDerivative(
                dims=par["nz"],
                axis=0,
                sampling=par["dz"],
                kind="centered",
                edge=par["edge"],
                order=order,
                dtype=par["dtype"],
            )
            assert Fop_MPI.shape == Fop.shape
            y_np = Fop @ x_global.get()
            y_adj_np = Fop.H @ x_global.get()
            assert_allclose(y.get(), y_np, rtol=1e-14)
            assert_allclose(y_adj.get(), y_adj_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1b), (par1j), (par1e), (par2), (par2b),
                                 (par2j), (par2e), (par3), (par3b), (par3j), (par3e),
                                 (par4), (par4b), (par4j), (par4e)])
def test_second_derivative_forward_nccl(par):
    """MPISecondDerivative operator (forward stencil)"""
    Sop_MPI = pylops_mpi.basicoperators.MPISecondDerivative(
        dims=par["nz"],
        sampling=par["dz"],
        kind="forward",
        edge=par["edge"],
        dtype=par["dtype"],
    )
    x = pylops_mpi.DistributedArray(
        global_shape=np.prod(par["nz"]),
        base_comm_nccl=nccl_comm,
        dtype=par["dtype"],
        partition=par["partition"],
        engine="cupy",
    )
    x[:] = cp.random.normal(rank, 10, x.local_shape)
    x_global = x.asarray()
    # Forward
    y_dist = Sop_MPI @ x
    y = y_dist.asarray()
    # Adjoint
    y_adj_dist = Sop_MPI.H @ x
    y_adj = y_adj_dist.asarray()
    # Dot test
    dottest(Sop_MPI, x, y_dist, np.prod(par["nz"]), np.prod(par["nz"]))

    if rank == 0:
        Sop = pylops.SecondDerivative(
            dims=par["nz"],
            axis=0,
            sampling=par["dz"],
            kind="forward",
            edge=par["edge"],
            dtype=par["dtype"],
        )
        assert Sop_MPI.shape == Sop.shape
        y_np = Sop @ x_global.get()
        y_adj_np = Sop.H @ x_global.get()
        assert_allclose(y.get(), y_np, rtol=1e-14)
        assert_allclose(y_adj.get(), y_adj_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1b), (par1j), (par1e), (par2), (par2b),
                                 (par2j), (par2e), (par3), (par3b), (par3j), (par3e),
                                 (par4), (par4b), (par4j), (par4e)])
def test_second_derivative_backward_nccl(par):
    """MPISecondDerivative operator (backward stencil)"""
    Sop_MPI = pylops_mpi.basicoperators.MPISecondDerivative(
        dims=par["nz"],
        sampling=par["dz"],
        kind="backward",
        edge=par["edge"],
        dtype=par["dtype"],
    )
    x = pylops_mpi.DistributedArray(
        global_shape=np.prod(par["nz"]),
        base_comm_nccl=nccl_comm,
        dtype=par["dtype"],
        partition=par["partition"],
        engine="cupy",
    )
    x[:] = cp.random.normal(rank, 10, x.local_shape)
    x_global = x.asarray()
    # Forward
    y_dist = Sop_MPI @ x
    y = y_dist.asarray()
    # Adjoint
    y_adj_dist = Sop_MPI.H @ x
    y_adj = y_adj_dist.asarray()
    # Dot test
    dottest(Sop_MPI, x, y_dist, np.prod(par["nz"]), np.prod(par["nz"]))

    if rank == 0:
        Sop = pylops.SecondDerivative(
            dims=par["nz"],
            axis=0,
            sampling=par["dz"],
            kind="backward",
            edge=par["edge"],
            dtype=par["dtype"],
        )
        assert Sop_MPI.shape == Sop.shape
        y_np = Sop @ x_global.get()
        y_adj_np = Sop.H @ x_global.get()
        assert_allclose(y.get(), y_np, rtol=1e-14)
        assert_allclose(y_adj.get(), y_adj_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par1), (par1b), (par1j), (par1e), (par2), (par2b),
                                 (par2j), (par2e), (par3), (par3b), (par3j), (par3e),
                                 (par4), (par4b), (par4j), (par4e)])
def test_second_derivative_centered_nccl(par):
    """MPISecondDerivative operator (centered stencil)"""
    Sop_MPI = pylops_mpi.basicoperators.MPISecondDerivative(
        dims=par["nz"],
        sampling=par["dz"],
        kind="centered",
        edge=par["edge"],
        dtype=par["dtype"],
    )
    x = pylops_mpi.DistributedArray(
        global_shape=np.prod(par["nz"]),
        base_comm_nccl=nccl_comm,
        dtype=par["dtype"],
        partition=par["partition"],
        engine="cupy",
    )
    x[:] = cp.random.normal(rank, 10, x.local_shape)
    x_global = x.asarray()
    # Forward
    y_dist = Sop_MPI @ x
    y = y_dist.asarray()
    # Adjoint
    y_adj_dist = Sop_MPI.H @ x
    y_adj = y_adj_dist.asarray()
    # Dot test
    dottest(Sop_MPI, x, y_dist, np.prod(par["nz"]), np.prod(par["nz"]))

    if rank == 0:
        Sop = pylops.SecondDerivative(
            dims=par["nz"],
            axis=0,
            sampling=par["dz"],
            kind="centered",
            edge=par["edge"],
            dtype=par["dtype"],
        )
        assert Sop_MPI.shape == Sop.shape
        y_np = Sop @ x_global.get()
        y_adj_np = Sop.H @ x_global.get()
        assert_allclose(y.get(), y_np, rtol=1e-14)
        assert_allclose(y_adj.get(), y_adj_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par5), (par5e), (par6), (par6e)])
def test_laplacian_nccl(par):
    """MPILaplacian Operator"""
    for kind in ["forward", "backward", "centered"]:
        Lop_MPI = pylops_mpi.basicoperators.MPILaplacian(
            dims=par["n"],
            axes=par["axes"],
            weights=par["weights"],
            sampling=par["sampling"],
            kind=kind,
            edge=par["edge"],
            dtype=par["dtype"],
        )
        x = pylops_mpi.DistributedArray(
            global_shape=np.prod(par["n"]),
            base_comm_nccl=nccl_comm,
            dtype=par["dtype"],
            engine="cupy",
        )
        x[:] = cp.random.normal(rank, 10, x.local_shape)
        x_global = x.asarray()
        # Forward
        y_dist = Lop_MPI @ x
        y = y_dist.asarray()
        # Adjoint
        y_adj_dist = Lop_MPI.H @ x
        y_adj = y_adj_dist.asarray()
        # Dot test
        dottest(Lop_MPI, x, y_dist, np.prod(par["n"]), np.prod(par["n"]))

        if rank == 0:
            Lop = pylops.Laplacian(
                dims=par["n"],
                axes=par["axes"],
                weights=par["weights"],
                sampling=par["sampling"],
                kind=kind,
                edge=par["edge"],
                dtype=par["dtype"],
            )
            assert Lop_MPI.shape == Lop.shape
            y_np = Lop @ x_global.get()
            y_adj_np = Lop.H @ x_global.get()
            assert_allclose(y.get(), y_np, rtol=1e-14)
            assert_allclose(y_adj.get(), y_adj_np, rtol=1e-14)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [(par5), (par5e), (par6), (par6e)])
def test_gradient_nccl(par):
    """MPIGradient Operator"""
    for kind in ["forward", "backward", "centered"]:
        Gop_MPI = pylops_mpi.basicoperators.MPIGradient(
            dims=par["n"],
            sampling=par["sampling"],
            kind=kind,
            edge=par["edge"],
            dtype=par["dtype"],
        )
        x_fwd = pylops_mpi.DistributedArray(
            global_shape=np.prod(par["n"]),
            base_comm_nccl=nccl_comm,
            dtype=par["dtype"],
            engine="cupy",
        )
        x_fwd[:] = cp.random.normal(rank, 10, x_fwd.local_shape)
        x_global = x_fwd.asarray()

        # Forward
        y_dist = Gop_MPI @ x_fwd
        assert isinstance(y_dist, pylops_mpi.StackedDistributedArray)
        y = y_dist.asarray()

        # Adjoint
        x_adj_dist1 = pylops_mpi.DistributedArray(
            global_shape=int(np.prod(par["n"])),
            base_comm_nccl=nccl_comm,
            dtype=par["dtype"],
            engine="cupy",
        )
        x_adj_dist1[:] = cp.random.normal(rank, 10, x_adj_dist1.local_shape)
        x_adj_dist2 = pylops_mpi.DistributedArray(
            global_shape=int(np.prod(par["n"])),
            base_comm_nccl=nccl_comm,
            dtype=par["dtype"],
            engine="cupy",
        )
        x_adj_dist2[:] = cp.random.normal(rank, 20, x_adj_dist2.local_shape)
        x_adj_dist3 = pylops_mpi.DistributedArray(
            global_shape=int(np.prod(par["n"])),
            base_comm_nccl=nccl_comm,
            dtype=par["dtype"],
            engine="cupy",
        )
        x_adj_dist3[:] = cp.random.normal(rank, 30, x_adj_dist3.local_shape)
        x_adj = pylops_mpi.StackedDistributedArray(
            distarrays=[x_adj_dist1, x_adj_dist2, x_adj_dist3]
        )
        x_adj_global = x_adj.asarray()
        y_adj_dist = Gop_MPI.H @ x_adj
        assert isinstance(y_adj_dist, pylops_mpi.DistributedArray)
        y_adj = y_adj_dist.asarray()

        # Dot test
        dottest(
            Gop_MPI, x_fwd, y_dist, len(par["n"]) * np.prod(par["n"]), np.prod(par["n"])
        )

        if rank == 0:
            Gop = pylops.Gradient(
                dims=par["n"],
                sampling=par["sampling"],
                kind=kind,
                edge=par["edge"],
                dtype=par["dtype"],
            )
            assert Gop_MPI.shape == Gop.shape
            y_np = Gop @ x_global.get()
            y_adj_np = Gop.H @ x_adj_global.get()
            assert_allclose(y.get(), y_np, rtol=1e-14)
            assert_allclose(y_adj.get(), y_adj_np, rtol=1e-14)
