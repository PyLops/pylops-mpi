__all__ = ["benchmark",
           "mark",
           ]

import functools
import logging
import os
import time
from typing import Callable, Optional, List
from mpi4py import MPI

from pylops.utils import deps as pylops_deps  # avoid namespace crashes with pylops_mpi.utils
from pylops_mpi.utils import deps

cupy_message = pylops_deps.cupy_import("the benchmark module")
nccl_message = deps.nccl_import("the benchmark module")

if nccl_message is None and cupy_message is None:
    from pylops_mpi.utils._nccl import _nccl_sync
else:
    def _nccl_sync():
        pass

# Benchmark is enabled by default
ENABLE_BENCHMARK = int(os.getenv("BENCH_PYLOPS_MPI", 1)) == 1

# Stack of active mark functions for nested support
_mark_func_stack = []
_markers = []


def _parse_output_tree(markers: List[str]):
    """This function parses the list of strings gathered during the benchmark call and output them
    as one properly formatted string. The format of output string follows the hierarchy of function calls
    i.e., the nested funtion calls are indented.

    Parameters
    ----------
    markers: :obj:`list`, optional
        A list of markers/labels generated from the benchmark call
    """
    global _markers
    output = []
    stack = []
    i = 0
    while i < len(markers):
        label, time, level = markers[i]
        if label.startswith("[decorator]"):
            indent = "\t" * (level - 1)
            output.append(f"{indent}{label}: total runtime: {time:6f} s\n")
        else:
            if stack:
                prev_label, prev_time, prev_level = stack[-1]
                if prev_level == level:
                    indent = "\t" * level
                    output.append(f"{indent}{prev_label}-->{label}: {time - prev_time:6f} s\n")
                    stack.pop()

            # Push to the stack only if it is going deeper or still at the same level
            if i + 1 <= len(markers) - 1:
                _, _ , next_level = markers[i + 1]
                if next_level >= level:
                    stack.append(markers[i])
        i += 1
    # reset markers, allowing other benchmarked function to start fresh
    _markers = []
    return output


def _sync():
    """Synchronize all MPI processes or CUDA Devices"""
    _nccl_sync()
    MPI.COMM_WORLD.Barrier()


def mark(label: str):
    """This function allows users to measure time arbitary lines of the function

    Parameters
    ----------
    label: :obj:`str`
        A label of the mark. This signifies both 1) the end of the
        previous mark 2) the beginning of the new mark
    """
    if not ENABLE_BENCHMARK:
        return
    if not _mark_func_stack:
        raise RuntimeError("mark() called outside of a benchmarked region")
    _mark_func_stack[-1](label)


def benchmark(func: Optional[Callable] = None,
              description: Optional[str] = "",
              logger: Optional[logging.Logger] = None,
              ):
    """A wrapper for code injection for time measurement.

    This wrapper measures the start-to-end time of the wrapped function when
    decorated without any argument.

    It also allows users to put a call to mark() anywhere inside the wrapped function
    for fine-grain time benchmark. This wrapper defines the local_mark() and pushes it
    to the _mark_func_stack for isolation in case of nested call.
    The user-facing mark() will always call the function at the top of the _mark_func_stack.

    Parameters
    ----------
    func : :obj:`callable`, optional
        Function to be decorated. Defaults to ``None``.
    description : :obj:`str`, optional
        Description for the output text. Defaults to ``''``.
    logger: :obj:`logging.Logger`, optional
        A `logging.Logger` object for logging the benchmark text output. This logger must be setup before
        passing to this function to either writing output to a file or log to stdout. If `logger`
        is not provided, the output is printed to stdout.
    """

    def noop_decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped

    @functools.wraps(func)
    def decorator(func):
        def wrapper(*args, **kwargs):
            rank = MPI.COMM_WORLD.Get_rank()

            level = len(_mark_func_stack) + 1
            # The header is needed for later tree parsing. Here it is allocating its spot.
            # the tuple at this index will be replaced after elapsed time is calculated.
            _markers.append((f"[decorator]{description or func.__name__}", None, level))
            header_index = len(_markers) - 1

            def local_mark(label):
                _markers.append((label, time.perf_counter(), level))

            _mark_func_stack.append(local_mark)

            _sync()
            start_time = time.perf_counter()
            # the mark() called in wrapped function will now call local_mark
            result = func(*args, **kwargs)
            _sync()
            end_time = time.perf_counter()

            elapsed = end_time - start_time
            _markers[header_index] = (f"[decorator]{description or func.__name__}", elapsed, level)

            # In case of nesting, the wrapped callee must pop its closure from stack so that
            # when the callee returns, the wrapped caller operates on its closure (and its level label), which now becomes
            # the top of the stack.
            _mark_func_stack.pop()

            # all the calls have fininshed
            if not _mark_func_stack:
                if rank == 0:
                    output = _parse_output_tree(_markers)
                    if logger:
                        logger.info("".join(output))
                    else:
                        print("".join(output))
            return result
        return wrapper

    # The code still has to return decorator so that the in-place decorator with arguments
    # like @benchmark(logger=logger) does not throw the error and can be kept untouched.
    if not ENABLE_BENCHMARK:
        return noop_decorator if func is None else noop_decorator(func)

    return decorator if func is None else decorator(func)
