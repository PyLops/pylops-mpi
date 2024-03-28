.. _api:

PyLops MPI API
==============

The Application Programming Interface (API) of PyLops MPI enables distributed and parallel processing of
large-scale linear algebra operations, inversions and computations.


DistributedArray
----------------

.. currentmodule:: pylops_mpi

.. autosummary::
   :toctree: generated/

    Partition
    DistributedArray
    StackedDistributedArray

Linear operators
----------------

Templates
~~~~~~~~~

.. currentmodule:: pylops_mpi

.. autosummary::
   :toctree: generated/

    MPILinearOperator
    asmpilinearoperator
    MPIStackedLinearOperator

Basic Operators
~~~~~~~~~~~~~~~

.. currentmodule:: pylops_mpi.basicoperators

.. autosummary::
   :toctree: generated/

    MPIBlockDiag
    MPIStackedBlockDiag
    MPIVStack
    MPIStackedVStack
    MPIHStack

Derivatives
~~~~~~~~~~~

.. currentmodule:: pylops_mpi.basicoperators

.. autosummary::
   :toctree: generated/

    MPIFirstDerivative
    MPISecondDerivative
    MPILaplacian
    MPIGradient

Solvers
-------

Basic
~~~~~

.. currentmodule:: pylops_mpi.optimization.cls_basic

.. autosummary::
   :toctree: generated/

    CG
    CGLS

.. currentmodule:: pylops_mpi.optimization.basic

.. autosummary::
   :toctree: generated/

    cg
    cgls
