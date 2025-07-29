.. _benchmarkutility:

Benchmarking
============

PyLops-MPI users can convenienly benchmark the performance of their code with a simple decorator.
:py:func:`pylops_mpi.utils.benchmark` and :py:func:`pylops_mpi.utils.mark` support various
function calling patterns that may arise when benchmarking distributed code.

- :py:func:`pylops_mpi.utils.benchmark` is a **decorator** used to time the execution of entire functions.
- :py:func:`pylops_mpi.utils.mark` is a **function** used inside decorated functions to insert fine-grained time measurements.

.. note::
   This benchmark utility is enabled by default i.e., if the user decorates the function with :py:func:`@benchmark`, the function will go through
   the time measurements, adding overheads. Users can turn off the benchmark while leaving the decorator in-place with

   .. code-block:: bash

      >> export BENCH_PYLOPS_MPI=0

The usage can be as simple as:

.. code-block:: python

   @benchmark
   def function_to_time():
       # Your computation

The result will print out to the standard output.
For fine-grained time measurements, :py:func:`pylops_mpi.utils.mark` can be inserted in the code region of benchmarked functions:

.. code-block:: python

   @benchmark
   def funtion_to_time():
       # You computation that you may want to ignore it in benchmark
       mark("Begin Region")
       # You computation
       mark("Finish Region")

You can also nest benchmarked functions to track execution times across layers of function calls with the output being correctly formatted.
Additionally, the result can also be exported to the text file. For completed and runnable examples, visit :ref:`sphx_glr_tutorials_benchmarking.py`
