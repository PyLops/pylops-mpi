import functools
import logging
import sys
import time
from typing import Callable, Optional
from mpi4py import MPI


# TODO (tharitt): later move to env file or something
ENABLE_BENCHMARK = True

logging.basicConfig(level=logging.INFO, force=True)
# Stack of active mark functions for nested support
_mark_func_stack = []
_markers = []


def _parse_output_tree(markers: list[str]):
    output = []
    stack = []
    i = 0
    while i < len(markers):
        label, time, level = markers[i]
        if label.startswith("[header]"):
            output.append(f"{"\t" * (level - 1)}{label}: total runtime: {time:6f} s\n")
        else:
            if stack:
                prev_label, prev_time, prev_level = stack[-1]
                if prev_level == level:
                    output.append(f"{"\t" * level}{prev_label}-->{label}: {time - prev_time:6f} s\n")
                    stack.pop()

            # Push to the stack only if it is going deeper or still at the same level
            if i + 1 < len(markers) - 1:
                _, _ , next_level = markers[i + 1]
                if next_level >= level:
                    stack.append(markers[i])
        i += 1
    return output


def mark(label):
    """This function allows users to measure time arbitary lines of the function

    Parameters
    ----------
    label: :obj:`str`
        A label of the mark. This signifies both 1) the end of the
        previous mark 2) the beginning of the new mark
    """
    if not _mark_func_stack:
        raise RuntimeError("mark() called outside of a benchmarked region")
    _mark_func_stack[-1](label)


def benchmark(func: Optional[Callable] = None,
              description="",
              save_file=False,
              file_path='benchmark.log'
              ):
    """A wrapper for code injection for time measurement.

    This wrapper measure the start-to-end time of the wrapped function when
    decorated without any argument.
    It also allows users to put a call to mark() anywhere inside the wrapped function
    for fine-grain time benchmark. This wrapper defines the local_mark() and push it
    the the _mark_func_stack for isolation in case of nested call.
    The user-facing mark() will always call the function at the top of the _mark_func_stack.

    Parameters
    ----------
    func : :obj:`callable`, optional
        Function to be decorated. Defaults to ``None``.
    description : :obj:`str`, optional
        Description for the output text. Defaults to ``''``.
    save_file : :obj:`bool`, optional
        Flag for saving file to a disk. Otherwise, the result will output to stdout. Defaults to ``False``
    file_path : :obj:`str`, optional
        File path for saving the output. Defaults to ``benchmark.log``

    """

    # Zero-overhead
    if not ENABLE_BENCHMARK:
        return func

    @functools.wraps(func)
    def decorator(func):
        def wrapper(*args, **kwargs):
            rank = MPI.COMM_WORLD.Get_rank()

            level = len(_mark_func_stack) + 1
            # The header is needed for later tree parsing. Here it is allocating its spot.
            # the tuple at this index will be replaced after elapsed time is calculated.
            _markers.append((f"[header]{description or func.__name__}", None, level))
            header_index = len(_markers) - 1

            def local_mark(label):
                _markers.append((label, time.perf_counter(), level))

            _mark_func_stack.append(local_mark)

            start_time = time.perf_counter()
            # the mark() called in wrapped function will now call local_mark
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            elapsed = end_time - start_time
            _markers[header_index] = (f"[header]{description or func.__name__}", elapsed, level)

            # In case of nesting, the wrapped callee must pop its closure from stack so that
            # when the callee returns, the wrapped caller operates on its closure (and its level label), which now becomes
            # the top of the stack.
            _mark_func_stack.pop()

            # all the calls have fininshed
            if not _mark_func_stack:
                if rank == 0:
                    output = _parse_output_tree(_markers)
                    logger = logging.getLogger()
                    # remove the stdout
                    for h in logger.handlers[:]:
                        logger.removeHandler(h)
                    handler = logging.FileHandler(file_path, mode='w') if save_file else logging.StreamHandler(sys.stdout)
                    handler.setLevel(logging.INFO)
                    logger.addHandler(handler)
                    logger.info("".join(output))
                    logger.removeHandler(handler)
                    if save_file:
                        handler.close()

            return result
        return wrapper
    if func is not None:
        return decorator(func)

    return decorator
