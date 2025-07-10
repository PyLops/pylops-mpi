import functools
import time

# TODO (tharitt): later move to env file or something
ENABLE_BENCHMARK = True

# This function is to be instrumented throughout the targeted function
def mark(label):
    if _current_mark_func is not None:
        _current_mark_func(label)

# Global hook - this will be re-assigned (points to)
# the function defined in benchmark wrapper
_current_mark_func = None

def benchmark(func):
    """A wrapper for code injection for time measurement.

    This wrapper allows users to put a call to mark()
    anywhere inside the wrapped function. The function mark()
    is defined in the global scope to be a placeholder for the targeted
    function to import. This wrapper will make it points to local_mark() defined
    in this function. Therefore, the wrapped function will be able call 
    local_mark(). All the context for local_mark() like mark list can be
    hidden from users and thus provide clean interface.

    Parameters
    ----------
    func : :obj:`callable`, optional
        Function to be decorated.
    """

    # Zero-overhead
    if not ENABLE_BENCHMARK:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        marks = []

        # currently this simply record the user-define label and record time
        def local_mark(label):
            marks.append((label, time.perf_counter()))

        global _current_mark_func
        _current_mark_func = local_mark

        # the mark() called in wrapped function will now call local_mark
        result = func(*args, **kwargs)
        # clean up to original state
        _current_mark_func = None

        # TODO (tharitt): maybe changing to saving results to file instead 
        if marks:
            prev_label, prev_t = marks[0]
            print(f"[BENCH] {prev_label}: 0.000000s")
            for label, t in marks[1:]:
                print(f"[BENCH] {label}: {t - prev_t:.6f}s since '{prev_label}'")
                prev_label, prev_t = label, t
        return result

    return wrapper
