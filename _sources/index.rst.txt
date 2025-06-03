Overview
========
PyLops-MPI is a Python library built on top of `PyLops <https://pylops.readthedocs.io/en/stable/>`_, designed to enable distributed and
parallel processing of large-scale linear algebra operations and computations.

Linear operators and large-scale inverse problems are at the core of many of the most commonly used algorithms in signal
processing, image processing, and remote sensing.
Pylops-MPI represents a linear operator by functions which describe matrix-vector products in
both forward and adjoint modes. These operators conduct computations within a distributed MPI environment,
using the power of the `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ library to optimize performance and scalability.
PyLops-MPI also provides iterative solvers (e.g. cgls) for many different types of problems, in particular to perform inversions.

By integrating MPI (Message Passing Interface), PyLops-MPI optimizes the collaborative processing power of multiple
computing nodes, enabling large and intricate tasks to be divided, solved, and aggregated in an efficient and
parallelized manner.

PyLops-MPI also supports the Nvidia's Collective Communication Library `(NCCL) <https://developer.nvidia.com/nccl>`_ for high-performance
GPU-to-GPU communications. The PyLops-MPI's NCCL engine works congruently with MPI by delegating the GPU-to-GPU communication tasks to 
highly-optimized NCCL, while leveraging MPI for CPU-side coordination and orchestration.

Get started by :ref:`installing PyLops-MPI <Installation>` and following our quick tour.

Terminology
-----------
A central *class*, :py:class:`pylops_mpi.DistributedArray`, is utilized throughout the entire library. This class offers the capability
to partition a large Numpy Array into smaller local arrays, distributing them across various ranks. It also facilitates broadcasting
the Numpy Array to different processes. Serving as our foundational array class, this class provides an efficient alternative to
using only the Numpy Arrays directly, resulting in increased efficiency.

A common *terminology* is used within the entire documentation of PyLops-MPI. Every MPI Linear Operator and its application to
a model will be referred to as **forward model (or operation)**

.. math::
    \mathbf{y} =  \mathbf{A} \mathbf{x}

while its application to a data is referred to as **adjoint model (or operation)**

.. math::
    \mathbf{x} = \mathbf{A}^H \mathbf{y}

Here, :math:`\mathbf{x}` is referred to as the model, and :math:`\mathbf{y}` is referred to as the data.
Both :math:`\mathbf{x}` and :math:`\mathbf{y}` are instances of the :py:class:`pylops_mpi.DistributedArray` class.
The *operator* :math:`\mathbf{A}:\mathbb{F}^m \to \mathbb{F}^n` effectively maps a
vector of size :math:`m` in the *model space* to a vector of size :math:`n`
in the *data space*, conversely the *adjoint operator*
:math:`\mathbf{A}^H:\mathbb{F}^n \to \mathbb{F}^m` maps a
vector of size :math:`n` in the *data space* to a vector of size :math:`m`
in the *model space*. As linear operators mimics the effect a matrix on a vector
we can also loosely refer to :math:`m` as the number of *columns* and :math:`n` as the
number of *rows* of the operator.

Ultimately, solving inverse problems accounts to removing the effect of
:math:`\mathbf{A}` from the data :math:`\mathbf{y}` to retrieve the model :math:`\mathbf{x}`.

Implementation
--------------
Pylops-MPI provides a :py:class:`pylops_mpi.MPILinearOperator` which allows the creation of new objects/operators
for matrix-vector products that can ultimately be used to solve any inverse problem of the form
:math:`\mathbf{y}=\mathbf{A}\mathbf{x}`.

To construct a :py:class:`pylops_mpi.MPILinearOperator`, a user is required to pass appropriate arguments
to the constructor of this class, or subclass it. More specifically, the method ``_matvec`` must be implemented for
the *forward operator* and the method ``_rmatvec`` may be implemented to apply the *Hermitian adjoint*.
The attributes/properties ``dtype`` (may be None) and ``shape`` (pair of integers) must be provided during
``__init__`` of this class.

Any MPI linear operator developed within the PyLops-MPI library follows this philosophy. As explained more in detail in
:ref:`addingoperator` section, an MPI Linear Operator is created by subclassing the :py:class:`pylops_mpi.MPILinearOperator`
class and implementing the ``_matvec`` and ``_rmatvec``.

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    self
    installation.rst
    gpu.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference documentation

   api/index.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   gallery/index.rst
   tutorials/index.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting involved

   Implementing new operators  <adding.rst>
   Contributing <contributing.rst>
   Changelog <changelog.rst>
   Credits <credits.rst>
