.. _changelog:

Changelog
=========

Version 0.2.0
-------------

*Released on: 16/12/2024*

* Added support for using CuPy arrays with PyLops-MPI.
* Introduced the :class:`pylops_mpi.signalprocessing.MPIFredholm1` and :class:`pylops_mpi.waveeqprocessing.MPIMDC` operators.
* Allowed the ``UNSAFE_BROADCAST`` partition to give users an option to handle overflow in broadcast scenarios.
* Added a dottest function to perform dot tests on PyLops-MPI operators.
* Created a tutorial for Multi-Dimensional Deconvolution (MDD).

Version 0.1.0
-------------

*Released on: 13/04/2024*

* Adapted :func:`pylops_mpi.optimization.basic.cg` and :func:`pylops_mpi.optimization.basic.cgls` to handle :class:`pylops_mpi.StackedDistributedArray`.
* Added :class:`pylops_mpi.basicoperators.MPIGradient` operator.
* Added :class:`pylops_mpi.MPIStackedLinearOperator`, :class:`pylops_mpi.basicoperators.MPIStackedBlockDiag`, and :class:`pylops_mpi.basicoperators.MPIStackedVStack` operators.
* Added :class:`pylops_mpi.StackedDistributedArray`.


Version 0.0.1
-------------

*Released on: 28/08/2023*

* First official release of PyLops-MPI
