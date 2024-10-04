---
title: 'PyLops-MPI - MPI Powered PyLops with mpi4py'
tags:
  - Python
  - MPI
  - High Performance Computing
authors:
  - name: Rohan Babbar
    orcid: 0000-0002-7203-7641
    affiliation: 1
  - name: Matteo Ravasi
    orcid: 0000-0003-0020-2721
    affiliation: 2
  - name: Yuxi Hong
    orcid: 0000-0002-0741-6602
    affiliation: 3
affiliations:
  - name: Computer Science and Engineering, Cluster Innovation Center, University of Delhi, Delhi, India.
    index: 1
  - name: Earth Science and Engineering, Physical Sciences and Engineering (PSE), King Abdullah University of Science and Technology (KAUST), Thuwal, Kingdom of Saudi Arabia.
    index: 2
  - name: Postdoc Researcher (Computer Science), Lawrence Berkeley National Laboratory, Berkeley, California, United States of America.
    index: 3
date: 24 September 2024
bibliography: paper.bib
---

# Summary

Large-scale linear operations and inverse problems are fundamental to numerous algorithms in fields such as image
processing, geophysics, signal processing, and remote sensing. This paper presents PyLops-MPI, an extension of PyLops
designed for distributed and parallel processing of large-scale challenges. PyLops-MPI facilitates forward and adjoint
matrix-vector products, as well as inversion solvers, in a distributed framework. By using the Message Passing
Interface (MPI), this framework effectively utilizes the computational power of multiple nodes or processors, enabling
efficient solutions to large and complex inversion tasks in a parallelized manner.

# Statement of need

As scientific datasets grow and the demand for higher resolution increases, the need for distributed computing alongside
matrix-free linear algebra becomes more critical. The size of models and datasets often exceeds the memory capacity of a
single machine—making it difficult to perform computations efficiently and accurately. Many operators consist of
multiple computational blocks that are resource-intensive—that can be effectively parallelized, further emphasizing the
necessity for a distributed approach.

When addressing distributed inverse problems, we identify three distinct use cases that highlight the need for a
flexible, scalable framework:

- **Fully Distributed Models and Data**: Both the model and data are distributed across nodes, with minimal
  communication during the modeling process. Communication occurs mainly during the solver stage when dot 
  products or regularization, such as the Laplacian, are applied. In this scenario where each node
  handles a portion of the model and data, and communication only happens between the model and data at each node.

- **Distributed Data, Model Available on All Nodes**: In this case, data is distributed across nodes while the model is
  available at all nodes. Communication is required during the adjoint pass when models produced by each node need 
  to be summed, and in the solver when performing dot products on the data.

- **Model and Data Available on All Nodes or Master**: Here, communication is confined to the operator, with the master
  node distributing parts of the model or data to workers. The workers then perform computations without requiring 
  communication in the solver.

Recent updates to mpi4py (version 3.0 and above) [@Dalcin:2021] have simplified its integration, enabling more efficient data
communication between nodes and processes.
Some projects in the Python ecosystem, such as mpi4py-fft [@Mortensen:2019], mcdc [@Morgan:2024], and mpi4jax [@mpi4jax],
utilize MPI to extend its capabilities,
improving the efficiency and scalability of distributed computing.

PyLops-MPI is built on top of PyLops[@Ravasi:2020] and utilizes mpi4py to enable an efficient framework to deal with
large scale problems in a distributed and parallelized manner.
PyLops-MPI offers an intuitive API that allows users to easily scatter and broadcast data and models across different
nodes or processors, enabling matrix-vector and adjoint matrix-vector operations in a distributed manner. It provides a
suite of MPI Linear Operators (MPI Powered Linear Operators) and MPI-powered inversion solvers, along with the
flexibility to create custom solvers tailored to specific needs.

What sets PyLops-MPI apart from other libraries is its ease of use in creating MPI Operators, facilitating efficient
integration between mpi4py and PyLops. This enables users to solve large-scale, complex inverse problems without the
risk of data leaks or the need to manage MPI requirements themselves.

# Software Framework

PyLops-MPI introduces MPI support to PyLops by providing an efficient API for solving linear problems through
parallelization using the mpi4py library. This library is designed to tackle large-scale inverse linear problems that
are difficult to solve using a single process.

![Software Framework representation of the ``PyLops-MPI`` API.](figs/software_framework.png)

The main components of the library include:

## DistributedArray

The `pylops_mpi.DistributedArray` class serves as the fundamental array class used throughout the library. It enables
the
partitioning of large NumPy[@Harris:2020] or CuPy[@cupy] arrays into smaller local arrays, which can
be distributed across different ranks.
Additionally, it allows for broadcasting the NumPy or CuPy array to multiple processes.

The DistributedArray supports two types of partitions through the **partition** attribute: `Partition.SCATTER`
distributes
the data across all ranks, allowing users to specify how much load each rank should handle, while `Partition.BROADCAST`
creates a copy of the data and distributes it to all ranks, ensuring that the data is available on each rank.

Furthermore, various basic mathematical functions are implemented for operations using the DistributedArray:

- Add (+) / Subtract (-): Adds or subtracts two DistributedArrays.
- Multiply (*): Multiplies two DistributedArrays.
- Dot-product (@): Calculates the dot product by flattening the arrays, resulting in a scalar value.
- Conj: Computes the conjugate of the DistributedArray.
- Norms: Calculates the vector norm along any specified axis.
- Copy: Creates a deep copy of the DistributedArray.

PyLops-MPI also provides a way to stack a series of `pylops_mpi.DistributedArray` objects using the
`pylops_mpi.StackedDistributedArray` class and perform mathematical operations on them.

## MPILinearOperator

`pylops_mpi.MPILinearOperator` is the base class for all MPI linear operators, allowing users to create new operators
for matrix-vector products that can solve various inverse problems. To create a new MPILinearOperator, users need to
subclass the `pylops_mpi.MPILinearOperator` parent class and specify the **shape** and **dtype**. The **_matvec** method
should be implemented for the forward operator, and the **_rmatvec** method should be used for the Hermitian adjoint.

## MPIStackedLinearOperator

`pylops_mpi.MPIStackedLinearOperator` serves as the base class that allows users to create operators where they want to
stack MPI Linear Operators. If the MPIOperator includes stacking of `pylops_mpi.MPILinearOperator` objects, it must
subclass this class. The `pylops_mpi.MPIStackedLinearOperator` has the ability to perform matrix-vector products with
both `pylops_mpi.DistributedArray` and `pylops_mpi.StackedDistributedArray`. Similar to `pylops_mpi.MPILinearOperator`,
users need to subclass the `pylops_mpi.MPIStackedLinearOperator` parent class and specify the **shape** and **dtype**.
The **_matvec** method should be implemented for the forward operator, while the **_rmatvec** method should be used for
the Hermitian adjoint.

## MPI Powered Solvers

PyLops-MPI offers a range of MPI-powered solvers that tackle linear problems using a standard least-squares cost
function. These solvers leverage **DistributedArray** and **MPILinearOperators** to perform inversion calculations. Our
solvers can be found within the submodule `pylops_mpi.optimization`.

## Halo Exchange

PyLops-MPI Linear Operators typically utilize halo exchange to facilitate operations at each rank. We encourage users to
ensure that the local data shapes at each rank are consistent to enable matrix-vector products without requiring
external communication. However, if the local shapes of the data and model at each rank do not match during these
operations, the operator performs a halo exchange, transferring boundary data cells (commonly referred to as "ghost
cells") to/from neighboring processes. This process ensures that the model vector and data vector shapes at each rank
are correctly aligned for the operation. Consequently, this data transfer enables efficient local computations without
the need for explicit inter-process communication, thereby avoiding heavy communication overhead.

# Use Cases

- *Post Stack Inversion - 3D* - Post-stack inversion represents the quantitative characterization of the
  subsurface [@Ravasi:2021]. In 3D, both the post-stack linear model and the data are three-dimensional. PyLops-MPI
  solves this problem by distributing one of the axes across different ranks, allowing matrix-vector products and
  inversions to take place at each rank, which are later gathered to obtain the inverted model.

- *Least-Squares Seismic Migration (LSSM)* involves manipulating seismic data to create an image of subsurface
  reflectivity[@Nemeth:1999]. PyLops MPI breaks this problem by distributing the sources across different MPI ranks.
  Each rank applies the source modeling operator to perform matrix-vector products with the broadcasted reflectivity.
  The resulting data is then inverted using the MPI-Powered solvers to produce the desired subsurface image.

# References