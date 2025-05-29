__all__ = [
    "nccl_enabled"
]

import os
from importlib import import_module, util
from typing import Optional


# error message at import of available package
def nccl_import(message: Optional[str] = None) -> str:
    nccl_test = (
        # detect if nccl is available and the user is expecting it to be used
        # CuPy must be checked first otherwise util.find_spec assumes it presents and check nccl immediately and lead to crash
        util.find_spec("cupy") is not None and util.find_spec("cupy.cuda.nccl") is not None and int(os.getenv("NCCL_PYLOPS_MPI", 1)) == 1
    )
    if nccl_test:
        # try importing it
        try:
            import_module("cupy.cuda.nccl")  # noqa: F401

            # if succesful, set the message to None
            nccl_message = None
        # if unable to import but the package is installed
        except (ImportError, ModuleNotFoundError) as e:
            nccl_message = (
                f"Fail to import cupy.cuda.nccl, Falling back to pure MPI (error: {e})."
                "Please ensure your CUDA NCCL environment is set up correctly "
                "for more detials visit 'https://docs.cupy.dev/en/stable/install.html'"
            )
            print(UserWarning(nccl_message))
    else:
        nccl_message = (
            "cupy.cuda.nccl package not installed or os.getenv('NCCL_PYLOPS_MPI') == 0. "
            f"In order to be able to use {message} "
            "ensure 'os.getenv('NCCL_PYLOPS_MPI') == 1'"
            "for more details for installing NCCL visit 'https://docs.cupy.dev/en/stable/install.html'"
        )

    return nccl_message


nccl_enabled: bool = (
    True if (nccl_import() is None and int(os.getenv("NCCL_PYLOPS_MPI", 1)) == 1) else False
)
