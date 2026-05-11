# 0.6.0
* Introduced `pylops_mpi.optimization.eigs.power_iteration` to estimate the maximum eigenvalue.
* Added `pylops_mpi.optimization.sparsity.ista` and the corresponding `pylops_mpi.optimization.cls_sparsity.ISTA` class, implementing the iterative shrinkage-thresholding algorithm.
* Added `pylops_mpi.optimization.sparsity.fista` and the corresponding `pylops_mpi.optimization.cls_sparsity.FISTA` class, a fast variant of ISTA.
* Included tutorial: "Reflectivity Inversion - 3D", with separate implementations for base, CuPy, and NCCL.
* Added `empty_like` function to `pylops_mpi.DistributedArray` and `pylops_mpi.StackedDistributedArray`.
* Introduced a `vdot` parameter in the `dot` method of `pylops_mpi.DistributedArray` and `pylops_mpi.StackedDistributedArray` to support vector dot products.

# 0.5.0
* Performed fixes to support `numpy>=2.4`.
* Added `redistribute` function to `pylops_mpi.DistributedArray`.
* Introduced MPI_Allgatherv in `pylops_mpi.utils._mpi.mpi_allgather` for variable sized arrays.
* Modified `pylops_mpi.DistributedArray.norm` to correctly handle cases where distributed axis differs from norm axis.

# 0.4.0
* Added `pylops_mpi.Distributed.DistributedMixIn` class with
  communicator-agnostic calls to communication methods.
* Added `pylops_mpi.utils._mpi` with implementations of MPI
  communication methods.
* Added `kind="summa"` implementation in 
  `pylops_mpi.basicoperators.MPIMatrixMult` operator.
* Added `kind` paramter to all operators in `pylops_mpi.basicoperators.MPILaplacian`
* Added `cp.cuda.Device().synchronize()` before any MPI call when using
  Cuda-Aware MPI.
* Modified `pylops_mpi.utils._nccl.initialize_nccl_comm` to 
  handle nodes with more GPUs than ranks.
* Fixed bug in `pylops_mpi.DistributedArray.__neg__` by
  explicitely passing `base_comm_nccl` during internal creation 
  of distributed array .

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