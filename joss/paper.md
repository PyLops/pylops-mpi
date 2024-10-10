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

As scientific datasets grow, the need for distributed computing and matrix-free linear algebra becomes crucial. 
Models and datasets often exceed a single machine’s memory, making efficient computation challenging. Many linear operators 
in scientific inverse problems can be decomposed into a series of computational blocks that are well-suited for parallelization, 
emphasizing the need for a distributed approach.

When addressing distributed inverse problems, we identify three distinct families of problems:

- **1. Fully distributed models and data**: Both the model and data are split across nodes with minimal communication, mainly 
  in the solver for dot products or regularization. Each node processes its own portion of the model and data.

- **2. Distributed data, model available on all nodes**: Data is distributed across nodes, but the model is available on all. 
  Communication happens during the adjoint pass to sum models and in the solver for data vector operations.

- **3. Model and data available on all nodes**: All nodes have identical copies of the data and model. Communication is limited 
  to operator calculations, with no communication in solver needed.

MPI for Python (mpi4py [@Dalcin:2021]) provides Python bindings for the MPI standard, allowing applications to leverage multiple 
processors across workstations, clusters, and supercomputers. Projects like mpi4py-fft [@Mortensen:2019], mcdc [@Morgan:2024], and mpi4jax [@mpi4jax] 
utilize mpi4py to expand their distributed computing capabilities, improving efficiency and scalability.

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

The `pylops_mpi.DistributedArray` class serves as the fundamental array class, enabling partitioning of large 
NumPy [@Harris:2020] or CuPy [@cupy] arrays into smaller local arrays distributed across different ranks and supporting 
broadcasting these arrays to multiple processes. The DistributedArray provides two partition types: `Partition.SCATTER` and 
`Partition.BROADCAST`. It also supports basic math operations such as addition (+), multiplication (*), dot-product (@), and 
more. Additionally, DistributedArray objects can be stacked using `pylops_mpi.StackedDistributedArray` for further operations.

## HStack, VStack, BlockDiag Operators

`pylops_mpi.MPILinearOperator` and `pylops_mpi.MPIStackedLinearOperator` serve as the foundation for creating new MPI operators. 
All existing operators subclass one of these classes.

PyLops enables easy combinations of linear operators via i)horizontal, ii)vertical, and iii)diagonal stacking. PyLops-MPI provides 
distributed versions of these, like `pylops_mpi.MPIBlockDiag`, which runs multiple operators in parallel on separate portions of the model 
and data (family 1). `pylops_mpi.MPIVStack` applies multiple operators in parallel to the whole model, with its adjoint summing different 
parts of the data vector (family 2). `pylops_mpi.MPIHStack` is the adjoint of MPIVStack.

## Halo Exchange

PyLops-MPI Linear Operators use halo exchange to transfer model and data portions between ranks. Users should ensure consistent local data shapes to avoid extra communication during
matrix-vector products. If shapes differ, the operator exchanges boundary data ("ghost cells") between neighboring processes, aligning shapes for efficient local computations 
and minimizing overhead.

## MPI-powered Solvers

PyLops-MPI offers a small subset of PyLops linear solvers, which can deal with **DistributedArray** and **StackedDistributedArray** objects. These solvers utilize 
the mathematical operations implemented for these classes and call the operator's forward and adjoint passes. 

# Use Cases

We briefly discuss three use cases in geophysical inverse problems that correspond to the three previously mentioned families of problems. 
Specifically:

- *1. Seismic Post-Stack Inversion* represents an effective approach to quantitative characterization of the
  subsurface [@Ravasi:2021] from seismic data. In 3D applications, both the model and data are three-dimensional (2 spatial coordinates and depth/time). PyLops-MPI addresses this by 
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