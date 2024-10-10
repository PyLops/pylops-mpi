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
  - name: Lawrence Berkeley National Laboratory, Berkeley, California, United States of America.
    index: 3
date: 24 September 2024
bibliography: paper.bib
---

# Summary

Linear algebra operations and inverse problems are fundamental to many algorithms in fields like image processing, geophysics, 
signal processing, and remote sensing. This paper introduces PyLops-MPI, an extension of the PyLops framework designed for 
distributed computing to solve large-scale inverse problems in Python. PyLops-MPI enables the parallelization of any combination 
of PyLops operators with various reduction patterns, provides drop-in replacements for certain operators to facilitate distributed
computing, and offers distributed solvers. By leveraging the Message Passing Interface (MPI) standard, this library effectively 
harnesses the computational power of multiple nodes, allowing users to scale their inversion problems efficiently with minimal 
changes to their single-node PyLops code.

# Statement of need

As scientific datasets grow and demand higher resolution, need for distributed computing and matrix-free linear algebra becomes essential. 
Models and datasets now often exceed a single machine's memory, making efficient, accurate computation challenging. However, 
many linear operators in scientific inverse problems can be decomposed into a series of computational blocks that, though 
resource-intensive, are well-suited for parallelization, highlighting the need for a distributed approach.

When addressing distributed inverse problems, we identify three distinct families of problems:

- **1. Fully distributed models and data**: Both model and data are distributed across nodes with minimal communication during 
  modeling, mainly occurring in the solver for dot products or regularization (e.g., Laplacian). Each node handles a portion 
  of the model and data when applying the operator and its adjoint.

- **2. Distributed data, model available on all nodes**: In this case, data is distributed across nodes while the model is
  available across all. Communication occurs during the adjoint pass when models produced by each node need
  to be summed together, and in the solver for dot products on the data vector.

- **3. Model and data available on all nodes**: Here, communication is limited to the operator, with nodes having identical
  copies of the data and model master. All nodes perform computations in the forward and adjoint passes of the operator, requiring
  no communication in the solver.

MPI for Python (mpi4py [@Dalcin:2021]) provides Python bindings for the MPI standard, allowing applications to leverage multiple 
processors across workstations, clusters, and supercomputers. Recent updates (version 3.0 and above) have simplified usage and improved 
data communication efficiency between nodes. Projects like mpi4py-fft [@Mortensen:2019], mcdc [@Morgan:2024], and mpi4jax [@mpi4jax] utilize mpi4py 
to expand their distributed computing capabilities, improving efficiency and scalability.

PyLops-MPI, built on top of PyLops [@Ravasi:2020], leverages mpi4py to address large-scale problems in a distributed and parallel manner. It provides an 
intuitive API for scattering and broadcasting data and models across nodes, allowing various mathematical operations (e.g., summation, subtraction, norms) 
to be performed. Additionally, it offers a suite of MPI-powered linear operators and solvers, with a flexible design for easy integration of custom operators
and solvers.

In summary, PyLops-MPI stands out for its ease of creating MPI Operators, facilitating efficient integration between mpi4py and PyLops, enabling users to 
solve complex inverse problems without worrying about data leaks or managing MPI requirements.

# Software Framework

PyLops-MPI is designed to tackle large-scale linear inverse problems that are difficult to solve using a single process 
(due to either extremely high computational cost or memory requirements).

![Software Framework representation of the ``PyLops-MPI`` API.](figs/software_framework.png)

The main components of the library include:

## DistributedArray

The `pylops_mpi.DistributedArray` class serves as the fundamental array class used throughout the library, enabling 
partitioning of large NumPy [@Harris:2020] or CuPy [@cupy] arrays into smaller local arrays distributed across different ranks 
and supporting broadcasting these arrays to multiple processes.

The DistributedArray supports two partition types via the **partition** attribute: `Partition.SCATTER` distributes data across all ranks 
with user-defined load, while `Partition.BROADCAST` creates a copy of the data for all ranks. Furthermore, various basic mathematical functions 
can be applied to DistributedArray objects, including addition (+), subtraction (-), multiplication (*), dot product (@), conjugate (Conj), vector norms, and 
deep copying (Copy). Additionally, users can stack `pylops_mpi.DistributedArray` objects using the `pylops_mpi.StackedDistributedArray` class for further mathematical 
operations.

## MPILinearOperator and MPIStackedLinearOperator

`pylops_mpi.MPILinearOperator` is the base class for creating MPI linear operators that perform matrix-vector products on DistributedArray objects.

`pylops_mpi.MPIStackedLinearOperator` represents a second level of abstraction in the creation of MPI-powered linear operators; allowing users to stack MPILinearOperator objects, 
enabling execution in a distributed manner and supporting matrix-vector products with both DistributedArray and StackedDistributedArray.

## HStack, VStack, BlockDiag Operators

One of PyLops' main features is the ability to create combinations of linear operators easily through three main design patterns:
i) horizontal stacking, ii) vertical stacking, and iii) diagonal stacking. PyLops-MPI offers distributed versions of these operators. Specifically, 
`pylops_mpi.MPIBlockDiag` enables multiple PyLops operators to run in parallel across different processes, each working on a portion of the model and 
data (see family 1). In contrast, `pylops_mpi.MPIVStack` runs multiple operators in parallel on the entire model in forward mode; its adjoint applies to different 
portions of the data vector, with individual outputs being sum-reduced (see family 2). Finally, `pylops_mpi.MPIHStack` is the adjoint of `pylops_mpi.MPIVStack`.

## Halo Exchange

PyLops-MPI Linear Operators typically use halo exchange to transfer portions of the model and data between consecutive ranks. While users should ensure 
consistent local data shapes across ranks for matrix-vector products without external communication, this may not always be feasible. When local shapes 
differ, the operator performs a halo exchange, transferring boundary data cells (or "ghost cells") to and from neighboring processes. This alignment of 
model and data vector shapes at each rank allows efficient local computations without explicit inter-process communication, minimizing communication overhead.

## MPI-powered Solvers

PyLops-MPI offers a small subset of PyLops linear solvers, which can deal with **DistributedArray** and **StackedDistributedArray** objects. These solvers utilize 
the mathematical operations implemented for these classes and call the operator's forward and adjoint passes. 

# Use Cases

We briefly discuss three use cases in geophysical inverse problems that correspond to the three previously mentioned families of problems. 
Specifically:

- *1. Seismic Post-Stack Inversion* represents an effective approach to quantitative characterization of the
  subsurface [@Ravasi:2021] from seismic data.  In 3D applications, both the model and data are three-dimensional (2 spatial coordinates and depth/time). PyLops-MPI addresses this by 
  distributing one spatial axis across different ranks, enabling matrix-vector products and inversions at each rank, which are then gathered to obtain the inverted model. 
  Communication typically occurs due to the introduction of regularization terms that promote smooth or blocky solutions.

- *2. Least-Squares Migration (LSM)* is the process of explaining seismic data via a Born modelling engine to produce an image of the subsurface
  reflectivity [@Nemeth:1999]. PyLops-MPI tackles this problem by distributing all the available sources across different MPI ranks.
  Each rank applies the expensive Born modeling operator for a subset of sources with the broadcasted reflectivity.
  The resulting data is scattered across the ranks, and inversion is performed with one of the MPI-powered solvers to generate the desired subsurface reflectivity image.

- *3. Multi-Dimensional Deconvolution (MDD)* is a powerful technique used at various stages of the seismic processing
  value chain to create datasets deprived of overburden effects [@Ravasi:2022]. PyLops-MPI tackles this large-scale inverse problem by distributing the kernel of the 
  Multi-Dimensional Deconvolution (MDC) operator across ranks, allowing each process to handle a portion of the batched matrix-vector (or matrix-matrix) multiplication required. 
  In this setup, both the model and data are accessible on all ranks throughout the inverse process.

Finally, we expect similar patterns to emerge in various inverse problems across different disciplines, which suggests a broad adoption of the PyLops-MPI framework in other 
scientific fields.

# Acknowledgements

The PyLops team acknowledges the support from Google Summer of Code and the NumFOCUS organization, which have been key to the development of PyLops-MPI.

# References