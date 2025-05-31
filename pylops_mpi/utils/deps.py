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
        # cupy.cuda.nccl comes with cupy installation so check the cupy
        util.find_spec("cupy") is not None and int(os.getenv("NCCL_PYLOPS_MPI", 1)) == 1
    )
    if nccl_test:
        # if cupy is present, this import will not throw error. The NCCL existence is checked with nccl.avaiable
        import cupy.cuda.nccl as nccl
        if nccl.available:
            # if succesfull, set the message to None
            nccl_message = None
        else:
            # if unable to import but the package is installed
            nccl_message = (
                        f"cupy is installed but cupy.cuda.nccl not available, Falling back to pure MPI."
                        "Please ensure your CUDA NCCL environment is set up correctly "
                        "for more details visit 'https://docs.cupy.dev/en/stable/install.html'"
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
