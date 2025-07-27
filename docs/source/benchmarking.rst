.. _benchmarkutility:

Benchmark Utility in PyLops-MPI
===============================
PyLops-MPI users can convenienly benchmark the performance of their code with a simple decorator.

This tutorial demonstrates how to use the :py:func:`pylops_mpi.utils.benchmark` and
:py:func:`pylops_mpi.utils.mark` utility methods in PyLops-MPI. These utilities support various
function calling patterns that may arise when benchmarking distributed code.

- :py:func:`pylops_mpi.utils.benchmark` is a **decorator** used to time the execution of entire functions.
- :py:func:`pylops_mpi.utils.mark` is a **function** used inside decorated functions to insert fine-grained time measurements.

Basic Setup
-----------

We start by importing the required modules and setting up some parameters of our simple program.

.. code-block:: python

   import sys
   import logging
   import numpy as np
   from mpi4py import MPI
   from pylops_mpi import DistributedArray, Partition

   from pylops_mpi.utils.benchmark import benchmark, mark

   np.random.seed(42)
   rank = MPI.COMM_WORLD.Get_rank()

   par = {'global_shape': (500, 501),
          'partition': Partition.SCATTER, 'dtype': np.float64,
          'axis': 1}

Benchmarking a Simple Function
------------------------------

We define a simple function and decorate it with :py:func:`benchmark`.

.. code-block:: python

   @benchmark
   def inner_func(par):
       dist_arr = DistributedArray(global_shape=par['global_shape'],
                                   partition=par['partition'],
                                   dtype=par['dtype'], axis=par['axis'])
       # may perform computation here
       dist_arr.dot(dist_arr)

Calling the function will result in the elapsed runtime being printed to standard output.

.. code-block:: python

   inner_func(par)

You can also customize the label of the printout using the ``description`` parameter:

.. code-block:: python

   @benchmark(description="printout_name")
   def my_func(...):
       ...

Fine-grained Time Measurements
------------------------------

To gain more insight into the runtime of specific code regions, use :py:func:`mark` within
a decorated function. This allows insertion of labeled time checkpoints.

.. code-block:: python

   @benchmark
   def inner_func_with_mark(par):
       mark("Begin array constructor")
       dist_arr = DistributedArray(global_shape=par['global_shape'],
                                   partition=par['partition'],
                                   dtype=par['dtype'], axis=par['axis'])
       mark("Begin dot")
       dist_arr.dot(dist_arr)
       mark("Finish dot")

The output will now contain timestamped entries for each marked location, along with the total time
from the outer decorator (marked with ``[decorator]`` in the output).

.. code-block:: python

   inner_func_with_mark(par)

Nested Function Benchmarking
----------------------------

You can nest benchmarked functions to track execution times across layers of function calls.
Below, we define an :py:func:`outerfunc_with_mark` that calls :py:func:`inner_func_with_mark` defined earlier.

.. code-block:: python

   @benchmark
   def outer_func_with_mark(par):
       mark("Outer func start")
       inner_func_with_mark(par)
       dist_arr = DistributedArray(global_shape=par['global_shape'],
                                   partition=par['partition'],
                                   dtype=par['dtype'], axis=par['axis'])
       dist_arr + dist_arr
       mark("Outer func ends")

Calling the function prints the full call tree with indentation, capturing both outer and nested timing.

.. code-block:: python

   outer_func_with_mark(par)

Logging Benchmark Output
------------------------

To store benchmarking results in a file, pass a custom :py:class:`logging.Logger` instance
to the :py:func:`benchmark` decorator. Below is a utility function that constructs such a logger.

.. code-block:: python

   def make_logger(save_file=False, file_path=''):
       logger = logging.getLogger(__name__)
       logging.basicConfig(filename=file_path if save_file else None,
                           filemode='w', level=logging.INFO, force=True)
       logger.propagate = False
       if save_file:
           handler = logging.FileHandler(file_path, mode='w')
       else:
           handler = logging.StreamHandler(sys.stdout)
       logger.addHandler(handler)
       return logger

Use this logger when decorating your function:

.. code-block:: python

   save_file = True
   file_path = "benchmark.log"
   logger = make_logger(save_file, file_path)

   @benchmark(logger=logger)
   def inner_func_with_logger(par):
       dist_arr = DistributedArray(global_shape=par['global_shape'],
                                   partition=par['partition'],
                                   dtype=par['dtype'], axis=par['axis'])
       # may perform computation here
       dist_arr.dot(dist_arr)

Run the function to generate output written directly to ``benchmark.log``.

.. code-block:: python

   inner_func_with_logger(par)

Final Notes
-----------

This tutorial demonstrated how to benchmark distributed PyLops-MPI operations using both
coarse and fine-grained instrumentation tools. These utilities help track and debug
performance bottlenecks in parallel workloads.

