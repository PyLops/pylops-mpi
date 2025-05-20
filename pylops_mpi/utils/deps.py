__all__ = [
    "nccl_enabled"
]

import os
from importlib import import_module, util
from typing import Optional


# error message at import of available package
def nccl_import(message: Optional[str] = None) -> str:
    nccl_test = (
        # TODO: setting of OS env should go to READme somewhere
        util.find_spec("cupy.cuda.nccl") is not None and int(os.getenv("NCCL_PYLOPS_MPI", 1)) == 1
    )
    if nccl_test:
        try:
            import_module("cupy.cuda.nccl")  # noqa: F401
            nccl_message = None
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
            "ensure 'os.getenv('NCCL_PYLOPS_MPI') == 1' and run "
            "<TO BE DECIDED>"
            "for more details visit 'https://docs.cupy.dev/en/stable/install.html'"
        )

    return nccl_message


nccl_enabled: bool = (
    True if (nccl_import() is None and int(os.getenv("NCCL_PYLOPS_MPI", 1)) == 1) else False
)
