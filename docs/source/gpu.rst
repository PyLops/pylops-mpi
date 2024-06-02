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

.. note::

   The CuPy backend is in active development, with many examples not yet in the docs.
   You can find many `other examples <https://github.com/PyLops/pylops_notebooks/tree/master/developement-mpi/Cupy_MPI>`_ from the `PyLops Notebooks repository <https://github.com/PyLops/pylops_notebooks>`_.