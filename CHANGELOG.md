
# 0.3.0
* Added `pylops_mpi.basicoperators.MPIMatrixMult` operator.
* Added NCCL support to all operators in :mod:`pylops_mpi.basicoperators`, 
  and  `pylops_mpi.signalprocessing`.
* Added ``base_comm_nccl`` in constructor of `pylops_mpi.DistributedArray`,
  to enable NCCL communication backend.
* Added `pylops_mpi.utils.benchmark` subpackage providing methods
  to decorate and mark functions / class methods to measure their execution 
  time.
* Added `pylops_mpi.utils._nccl` subpackage implementing methods
  for NCCL communication backend.
* Added `pylops_mpi.utils.deps` subpackage to safely import ``nccl``
* Fixed partition in the creation of the output distributed array in 
  `pylops_mpi.signalprocessing.MPIFredholm1`.

# 0.2.0
- Added support for using CuPy arrays with PyLops-MPI.
- Introduced the `pylops_mpi.signalprocessing.MPIFredholm1` and `pylops_mpi.waveeqprocessing.MPIMDC` operators.
- Allowed the `UNSAFE_BROADCAST` partition to give users an option to handle overflow in broadcast scenarios.
- Added a dottest function to perform dot tests on PyLops-MPI operators.
- Created a tutorial for Multi-Dimensional Deconvolution (MDD).

# 0.1.0
- Adapted `pylops_mpi.optimization.cg` and `pylops_mpi.optimization.cgls` to handle `pylops_mpi.StackedDistributedArray`.
- Added `pylops_mpi.MPIGradient` operator.
- Added `pylops_mpi.MPIStackedLinearOperator`, `pylops_mpi.MPIStackedBlockDiag`, and `pylops_mpi.MPIStackedVStack` operators.
- Added `pylops_mpi.StackedDistributedArray`.

# 0.0.1
- First official release of PyLops-MPI.