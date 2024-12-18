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