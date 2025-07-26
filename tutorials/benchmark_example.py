r"""
Benchmark Utility in PyLops-MPI
=========================
This tutorial demonstrates how to use the bencmark utility of PyLops-MPI. It contains various
function calling pattern that may come up during the benchmarking.
"""
import numpy as np
from mpi4py import MPI
from pylops_mpi import DistributedArray, Partition

np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()

par = {'global_shape': (500, 501),
       'partition': Partition.SCATTER, 'dtype': np.float64,
       'axis': 1}

###############################################################################
# Let's start by import the utility
from pylops_mpi.utils.benchmark import benchmark, mark

###############################################################################
# :py:func:`pylops_mpi.utils.benchmark` is a decorator used to decorate any
# function to measure its execution time from start to finish
# :py:func:`pylops_mpi.utils.mark` is a function used inside the benchmark-decorated
# function to provide fine-grain time measurements. Let's start with a simple example


@benchmark
def inner_func(par):
    dist_arr = DistributedArray(global_shape=par['global_shape'],
                                partition=par['partition'],
                                dtype=par['dtype'], axis=par['axis'])
    # may perform computation here
    dist_arr.dot(dist_arr)


###############################################################################
# When we call :py:func:`inner_func`, we will see the result
# of the benchmark print to standard output. If we want to customize the
# function name in the printout, we can pass the parameter to the :py:func:`benchmark`
# i.e., :py:func:`@benchmark(description="printout_name")`

inner_func(par)

###############################################################################
# We may want to get the finer time measurement by timing the execution time from arbitary lines
# of code. :py:func:`pylops_mpi.utils.mark` provides such utitlity


@benchmark
def inner_func_with_mark(par):
    mark("Begin array constructor")
    dist_arr = DistributedArray(global_shape=par['global_shape'],
                                partition=par['partition'],
                                dtype=par['dtype'], axis=par['axis'])
    mark("Begin dot")
    dist_arr.dot(dist_arr)
    mark("Finish dot")


###############################################################################
# Now when we run, we get the detail time measurement. Noted that there is a tag
# [decorator] to the function name to distinguish between the start-to-end time measuredment of
# top-level function and those that comes from :py:func:`pylops_mpi.utils.mark`
inner_func_with_mark(par)

###############################################################################
# This utility also supports the nested functions. Let's define the outer function
# that internally calls decorated :py:func:`inner_func_with_mark`


@benchmark
def outer_func_with_mark(par):
    mark("Outer func start")
    inner_func_with_mark(par)
    dist_arr = DistributedArray(global_shape=par['global_shape'],
                                partition=par['partition'],
                                dtype=par['dtype'], axis=par['axis'])
    dist_arr + dist_arr
    mark("Outer func ends")


###############################################################################
# If we run :py:func:`outer_func_with_mark`, we get the time measurement nicely
# printout with the nested indentation to specify that nested calls.
outer_func_with_mark(par)


###############################################################################
# In some cases, we may want to write benchmark output to a text file.
# :py:func:`pylops_mpi.utils.benchmark` also takes the py:class:`logging.Logger`
# in its argument. Let's first import the logging package and construct our logger

import sys
import logging
save_file = True
file_path = "benchmark.log"

###############################################################################
# Here we define a simple :py:func:`make_logger()`. We set the :py:func:`logger.propagate = False`
# isolate the logging of our benchmark from that of the rest of the code


def make_logger(save_file=False, file_path=''):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=file_path if save_file else None, filemode='w', level=logging.INFO, force=True)
    logger.propagate = False
    if save_file:
        handler = logging.FileHandler(file_path, mode='w')
    else:
        handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    return logger


logger = make_logger(save_file, file_path)


###############################################################################
# Then we can pass the logger to the :py:func:`pylops_mpi.utils.benchmark`

@benchmark(logger=logger)
def inner_func_with_logger(par):
    dist_arr = DistributedArray(global_shape=par['global_shape'],
                                partition=par['partition'],
                                dtype=par['dtype'], axis=par['axis'])
    # may perform computation here
    dist_arr.dot(dist_arr)


###############################################################################
# Run this function and observe that the file `benchmark.log` is written.
inner_func_with_logger(par)
