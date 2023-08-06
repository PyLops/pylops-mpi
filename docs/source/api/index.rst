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

Linear operators
----------------

Templates
~~~~~~~~~

.. currentmodule:: pylops_mpi

.. autosummary::
   :toctree: generated/

    MPILinearOperator
    asmpilinearoperator

Basic Operators
~~~~~~~~~~~~~~~

.. currentmodule:: pylops_mpi.basicoperators

.. autosummary::
   :toctree: generated/

    MPIBlockDiag
    MPIVStack
    MPIHStack

Derivatives
~~~~~~~~~~~

.. currentmodule:: pylops_mpi.basicoperators

.. autosummary::
   :toctree: generated/

    MPIFirstDerivative
    MPISecondDerivative

Solvers
-------

Basic
~~~~~

.. currentmodule:: pylops_mpi.optimization.cls_basic

.. autosummary::
   :toctree: generated/

    CGLS

.. currentmodule:: pylops_mpi.optimization.basic

.. autosummary::
   :toctree: generated/

    cgls
