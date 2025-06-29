.. _gpu:

GPU Support
===========

Overview
--------
PyLops-mpi supports computations on GPUs leveraging the GPU backend of PyLops. Under the hood,
`CuPy <https://cupy.dev/>`_ (``cupy-cudaXX>=v13.0.0``) is used to perform all of the operations.
This library must be installed *before* PyLops-mpi is installed.

.. note::

   Set environment variable ``CUPY_PYLOPS=0`` to force PyLops to ignore the ``cupy`` backend.
   This can be also used if a previous (or faulty) version of ``cupy`` is installed in your system,
   otherwise you will get an error when importing PyLops.


The :class:`pylops_mpi.DistributedArray` and :class:`pylops_mpi.StackedDistributedArray` objects can be 
generated using both ``numpy`` and ``cupy`` based local arrays, and all of the operators and solvers in PyLops-mpi 
can handle both scenarios. Note that, since most operators in PyLops-mpi are thin-wrappers around PyLops operators,
some of the operators in PyLops that lack a GPU implementation cannot be used also in PyLops-mpi when working with
cupy arrays.

Moreover, PyLops-MPI also supports the Nvidia's Collective Communication Library (NCCL) for highly-optimized
collective operations, such as AllReduce, AllGather, etc. This allows PyLops-MPI users to leverage the
proprietary technology like NVLink that might be available in their infrastructure for fast data communication.

.. note::

   Set environment variable ``NCCL_PYLOPS_MPI=0`` to explicitly force PyLops-MPI to ignore the ``NCCL`` backend.
   However, this is optional as users may opt-out for NCCL by skip passing `cupy.cuda.nccl.NcclCommunicator` to
   the :class:`pylops_mpi.DistributedArray` 

Example
-------

Finally, let's briefly look at an example. First we write a code snippet using
``numpy`` arrays which PyLops-mpi will run on your CPU:

.. code-block:: python

    # MPI helpers
    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    
    # Create distributed data (broadcast)
    nxl, nt = 20, 20
    dtype = np.float32
    d_dist = pylops_mpi.DistributedArray(global_shape=nxl * nt,                                   
                                         partition=pylops_mpi.Partition.BROADCAST,
                                         engine="numpy", dtype=dtype)
    d_dist[:] = np.ones(d_dist.local_shape, dtype=dtype)
    
    # Create and apply VStack operator
    Sop = pylops.MatrixMult(np.ones((nxl, nxl)), otherdims=(nt, ))
    HOp = pylops_mpi.MPIVStack(ops=[Sop, ])
    y_dist = HOp @ d_dist
    

Now we write a code snippet using ``cupy`` arrays which PyLops will run on 
your GPU:

.. code-block:: python

    # MPI helpers
    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    
    # Define gpu to use
    cp.cuda.Device(device=rank).use()

    # Create distributed data (broadcast)
    nxl, nt = 20, 20
    dtype = np.float32
    d_dist = pylops_mpi.DistributedArray(global_shape=nxl * nt,                                   
                                         partition=pylops_mpi.Partition.BROADCAST,
                                         engine="cupy", dtype=dtype)
    d_dist[:] = cp.ones(d_dist.local_shape, dtype=dtype)
    
    # Create and apply VStack operator
    Sop = pylops.MatrixMult(cp.ones((nxl, nxl)), otherdims=(nt, ))
    HOp = pylops_mpi.MPIVStack(ops=[Sop, ])
    y_dist = HOp @ d_dist

The code is almost unchanged apart from the fact that we now use ``cupy`` arrays,
PyLops-mpi will figure this out!

Finally, if NCCL is available, a ``cupy.cuda.nccl.NcclCommunicator`` can be initialized and passed to :class:`pylops_mpi.DistributedArray`
as follows:

.. code-block:: python

    from pylops_mpi.utils._nccl import initialize_nccl_comm

    # Initilize NCCL Communicator
    nccl_comm = initialize_nccl_comm()

    # Create distributed data (broadcast)
    nxl, nt = 20, 20
    dtype = np.float32
    d_dist = pylops_mpi.DistributedArray(global_shape=nxl * nt,
                                         base_comm_nccl=nccl_comm,
                                         partition=pylops_mpi.Partition.BROADCAST,
                                         engine="cupy", dtype=dtype)
    d_dist[:] = cp.ones(d_dist.local_shape, dtype=dtype)

    # Create and apply VStack operator
    Sop = pylops.MatrixMult(cp.ones((nxl, nxl)), otherdims=(nt, ))
    HOp = pylops_mpi.MPIVStack(ops=[Sop, ])
    y_dist = HOp @ d_dist

Under the hood, PyLops-MPI use both MPI Communicator and NCCL Communicator to manage distributed operations. Each GPU is logically binded to 
one MPI process. In fact, minor communications like those dealing with array-related shapes and sizes are still performed using MPI, while collective calls on array like AllReduce are carried through NCCL

.. note::

   The CuPy and NCCL backend is in active development, with many examples not yet in the docs.
   You can find many `other examples <https://github.com/PyLops/pylops_notebooks/tree/master/developement-mpi/Cupy_MPI>`_ from the `PyLops Notebooks repository <https://github.com/PyLops/pylops_notebooks>`_.

Supports for NCCL Backend
----------------------------
In the following, we provide a list of modules (i.e., operators and solvers) where we plan to support NCCL and the current status:

.. list-table::
   :widths: 50 25 
   :header-rows: 1

   * - modules
     - NCCL supported
   * - :class:`pylops_mpi.DistributedArray`
     - ✅ 
   * - :class:`pylops_mpi.basicoperators.MPIVStack`
     - ✅ 
   * - :class:`pylops_mpi.basicoperators.MPIVStack`
     - ✅ 
   * - :class:`pylops_mpi.basicoperators.MPIHStack`
     - ✅ 
   * - :class:`pylops_mpi.basicoperators.MPIBlockDiag`
     - ✅ 
   * - :class:`pylops_mpi.basicoperators.MPIGradient`
     - ✅ 
   * - :class:`pylops_mpi.basicoperators.MPIFirstDerivative`
     - ✅ 
   * - :class:`pylops_mpi.basicoperators.MPISecondDerivative`
     - ✅ 
   * - :class:`pylops_mpi.basicoperators.MPILaplacian`
     - ✅ 
   * - :class:`pylops_mpi.optimization.basic.cg`
     - ✅ 
   * - :class:`pylops_mpi.optimization.basic.cgls`
     - ✅ 
   * - :class:`pylops_mpi.signalprocessing.Fredhoml1`
     - ✅ 
   * - Complex Numeric Data Type for NCCL 
     - ✅ 
   * - ISTA Solver
     - Planned ⏳