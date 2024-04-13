.. _changelog:

Changelog
=========

Version 0.0.2
-------------

*Released on: 15/04/2024*

* Adapted :func:`pylops_mpi.optimization.basic.cg` and :func:`pylops_mpi.optimization.basic.cgls` to handle :class:`pylops_mpi.StackedDistributedArray`.
* Added :class:`pylops_mpi.basicoperators.MPIGradient` operator.
* Added :class:`pylops_mpi.MPIStackedLinearOperator`, :class:`pylops_mpi.basicoperators.MPIStackedBlockDiag`, and :class:`pylops_mpi.basicoperators.MPIStackedVStack` operators.
* Added :class:`pylops_mpi.StackedDistributedArray`.


Version 0.0.1
-------------

*Released on: 28/08/2023*

* First official release of PyLops-MPI
