import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cupy as cp
from pylops_mpi.utils.benchmark import benchmark, mark
from mpi4py import MPI

from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.lsm import LSM

import pylops_mpi
np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()

# Trade-off: model size is controlled by (nx, nz) which affects data transfer and
# (nr, ns) controls data size which affects the compute
par1 = {"nx": 30, "nz": 50, "ns": 9, "nr": 18, "use_cupy": False, "use_nccl": False}
par2 = {"nx": 30, "nz": 50, "ns": 9, "nr": 18, "use_cupy": True, "use_nccl": False}
par3 = {"nx": 30, "nz": 50, "ns": 9, "nr": 18, "use_cupy": True, "use_nccl": True}


def prepare_kirchhoff_op(par):
    v0 = 1500
    dx = 12.5
    dz = 4
    x, z = np.arange(par["nx"]) * dx, np.arange(par["nz"]) * dz

    # recv and source config
    rx = np.linspace(10 * dx, (par["nx"] - 10) * dx, par["nr"])
    rz = 20 * np.ones(par["nr"])
    recs = np.vstack((rx, rz))

    nstot = MPI.COMM_WORLD.allreduce(par["ns"], op=MPI.SUM)
    sxtot = np.linspace(dx * 10, (par["nx"] - 10) * dx, nstot)
    sx = sxtot[rank * par["ns"]: (rank + 1) * par["ns"]]
    sz = 10 * np.ones(par["ns"])
    sources = np.vstack((sx, sz))

    # Wavelet
    nt = 651
    dt = 0.004
    t = np.arange(nt) * dt
    wav, wavt, wavc = ricker(t[:41], f0=20)

    lsm_op = LSM(
        z,
        x,
        t,
        sources,
        recs,
        v0,
        cp.asarray(wav.astype(np.float32)) if par["use_cupy"] else wav,
        wavc,
        mode="analytic",
        engine="cuda" if par["use_cupy"] else "numba",
        dtype=np.float32
    )
    if par["use_cupy"]:
        lsm_op.Demop.trav_srcs = cp.asarray(lsm_op.Demop.trav_srcs.astype(np.float32))
        lsm_op.Demop.trav_recs = cp.asarray(lsm_op.Demop.trav_recs.astype(np.float32))

    return lsm_op


def prepare_distributed_data(par, lsm_op, nccl_comm):
    # Reflectivity Model
    refl = np.zeros((par["nx"], par["nz"]))
    refl[:, par["nz"] // 4] = -1
    refl[:, par["nz"] // 2] = 0.5
    refl_dist = pylops_mpi.DistributedArray(global_shape=par["nx"] * par["nz"],
                                            partition=pylops_mpi.Partition.BROADCAST,
                                            base_comm_nccl=nccl_comm,
                                            engine="cupy" if par["use_cupy"] else "numpy")
    refl_dist[:] = cp.asarray(refl.flatten()) if par["use_cupy"] else refl.flatten()

    VStack = pylops_mpi.MPIVStack(ops=[lsm_op.Demop, ])
    d_dist = VStack @ refl_dist
    return d_dist


@benchmark
def run_bench(par):
    # if run with MPI, NCCL should not be initialized at all to avoid hang
    if par["use_nccl"]:
        nccl_comm = pylops_mpi.utils._nccl.initialize_nccl_comm()
    else:
        nccl_comm = None

    mark(f"begin {par["use_cupy"]=}, {par["use_nccl"]=}")
    lsm_op = prepare_kirchhoff_op(par)
    d_dist = prepare_distributed_data(par, lsm_op, nccl_comm)
    VStack = pylops_mpi.MPIVStack(ops=[lsm_op.Demop, ])
    mark("after prepare")
    # TODO (tharitt): In the actual benchmark, we probably have to decorate
    # the matvec() and rmatvec() to separate computation from communication time
    madj_dist = VStack.H @ d_dist
    mark("after adjoint")
    _ = madj_dist.asarray().reshape((par["nx"], par["nz"]))


if __name__ == "__main__":
    run_bench(par1)
    print("========")
    run_bench(par2)
    print("========")
    run_bench(par3)
