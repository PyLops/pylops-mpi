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

    MPIMatrixMult
    MPIBlockDiag
    MPIStackedBlockDiag
    MPIVStack
    MPIStackedVStack
    MPIHStack
    MPIHalo
    

Derivatives
~~~~~~~~~~~

.. currentmodule:: pylops_mpi.basicoperators

.. autosummary::
   :toctree: generated/

    MPIFirstDerivative
    MPISecondDerivative
    MPILaplacian
    MPIGradient

Signal Processing
~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops_mpi.signalprocessing

.. autosummary::
   :toctree: generated/

    MPIFredholm1


Wave-Equation processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops_mpi.waveeqprocessing

.. autosummary::
   :toctree: generated/

    MPIMDC


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


Utils
-----

.. currentmodule:: pylops_mpi.DistributedArray

.. autosummary::
   :toctree: generated/

    local_split


.. currentmodule:: pylops_mpi.basicoperators.MatrixMult

.. autosummary::
   :toctree: generated/

    block_gather
    local_block_split
    active_grid_comm

.. currentmodule:: pylops_mpi.utils

.. autosummary::
   :toctree: generated/

    dottest
    benchmark
    mark