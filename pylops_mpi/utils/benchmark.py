import functools
import time
from typing import Callable, Optional
from mpi4py import MPI


# TODO (tharitt): later move to env file or something
ENABLE_BENCHMARK = True


# This function allows users to measure time arbitary lines of the function
def mark(label):
    if not _mark_func_stack:
        raise RuntimeError("mark() called outside of a benchmarked region")
    _mark_func_stack[-1](label)


# Stack of active mark functions for nested support
_mark_func_stack = []


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

            # Here we rely on the closure property of Python.
            # This marks will isolate from (shadow) the marks previously
            # defined in the function currently on top of the _mark_func_stack.
            marks = []

            def local_mark(label):
                marks.append((label, time.perf_counter()))
            _mark_func_stack.append(local_mark)

            start_time = time.perf_counter()
            # the mark() called in wrapped function will now call local_mark
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            _mark_func_stack.pop()

            output = []
            # start-to-end time
            elapsed = end_time - start_time

            # TODO (tharitt): Both MPI + NCCL collective calls have implicit synchronization inside the stream
            # So, output only from rank=0 should suffice. We can add per-rank output later on if makes sense.
            if rank == 0:
                level = len(_mark_func_stack)
                output.append(f"{'---' * level}{description or func.__name__} {elapsed:6f} seconds (rank = {rank})\n")
                if marks:
                    prev_label, prev_t = marks[0]
                    for label, t in marks[1:]:
                        output.append(f"{'---' * level}[{prev_label} ---> {label}]: {t - prev_t:.6f}s \n")
                        prev_label, prev_t = label, t

                if save_file:
                    with open(file_path, "a") as f:
                        f.write("".join(output))
                else:
                    print("".join(output))
            return result
        return wrapper
    if func is not None:
        return decorator(func)

    return decorator
