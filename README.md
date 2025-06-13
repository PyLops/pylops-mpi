![PyLops-MPI](https://github.com/PyLops/pylops-mpi/blob/main/docs/source/_static/pylopsmpi_b.png)

[![PyPI version](https://badge.fury.io/py/pylops-mpi.svg)](https://badge.fury.io/py/pylops-mpi)
[![Build status](https://github.com/PyLops/pylops-mpi/actions/workflows/build.yml/badge.svg)](https://github.com/PyLops/pylops-mpi/actions/workflows/build.yml)
[![Documentation status](https://github.com/PyLops/pylops-mpi/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/PyLops/pylops-mpi/actions/workflows/pages/pages-build-deployment)
![OS-support](https://img.shields.io/badge/OS-linux,osx-850A8B.svg)
[![Slack Status](https://img.shields.io/badge/chat-slack-green.svg)](https://pylops.slack.com)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07512/status.svg)](https://doi.org/10.21105/joss.07512)

# Distributed linear operators and solvers
Pylops-mpi is a Python library built on top of [PyLops](https://pylops.readthedocs.io/en/stable/), designed to enable distributed and parallel processing of 
large-scale linear algebra operations and computations.  

## Installation
To install pylops-mpi, you need to have Message Passing Interface (MPI) and optionally Nvidia's Collective Communication Library (NCCL) installed on your system.

1. **Download and Install MPI**: Visit the official MPI website to download an appropriate MPI implementation for your system. 
Follow the installation instructions provided by the MPI vendor.
   - [Open MPI](https://www.open-mpi.org/software/ompi/v1.10/)
   - [MPICH](https://www.mpich.org/downloads/)
   - [Intel MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html#gs.10j8fx)

2. **Verify MPI Installation**: After installing MPI, verify its installation by opening a terminal or command prompt 
and running the following command:
   ```
   mpiexec --version
   ```

3. **Install pylops-mpi**: Once MPI is installed and verified, you can proceed to install `pylops-mpi` via `pip`:
   ```
   pip install pylops-mpi
   ```

4. (Optional) To enable the NCCL backend for multi-GPU systems, install `cupy` and `nccl` via `pip`:
   ```
   pip install cupy-cudaXx nvidia-nccl-cuX
   ```
   
   with `X=11,12`.

Alternatively, if the Conda package manager is used to setup the Python environment, steps 1 and 2 can be skipped and install `mpi4py` which comes with its own MPI distribution:

```
conda install -c conda-forge mpi4py X
```

with `X=mpich, openmpi, impi_rt, msmpi`. Similarly step 4 can be accomplished using:

```
conda install -c conda-forge cupy nccl 
```

See the docs ([Installation](https://pylops.github.io/pylops-mpi/installation.html)) for more information.

## Run Pylops-MPI
Once you have installed the prerequisites and pylops-mpi, you can run pylops-mpi using the `mpiexec` command. 

Here is an example on how to run a python script called `<script_name>.py`:
```
mpiexec -n <NUM_PROCESSES> python <script_name>.py
```

## Example: A distributed finite-difference operator
The following example is a modified version of 
[PyLops' README](https://github.com/PyLops/pylops/blob/dev/README.md)_ starting 
example that can handle a 2D-array distributed across ranks over the first dimension 
via the `DistributedArray` object:

```python
import numpy as np
from pylops_mpi import DistributedArray, Partition

nx, ny = 11, 21
x = np.zeros((nx, ny), dtype=np.float64)
x[nx // 2, ny // 2] = 1.0

# Initialize  DistributedArray with partition set to Scatter
x_dist = pylops_mpi.DistributedArray.to_dist(
            x=x.flatten(), 
            partition=Partition.SCATTER)

# Distributed first-derivative
D_op = pylops_mpi.MPIFirstDerivative((nx, ny), dtype=np.float64)

# y = Dx
y_dist = D_op @ x

# xadj = D^H y
xadj_dist = D_op.H @ y_dist

# xinv = D^-1 y
x0_dist = pylops_mpi.DistributedArray(D_op.shape[1], dtype=np.float64)
x0_dist[:] = 0
xinv_dist = pylops_mpi.cgls(D_op, y_dist, x0=x0_dist, niter=10)[0]
```

Note that the `DistributedArray` class provides the `to_dist` class method that accepts a NumPy array as input and converts it into an instance of the `DistributedArray` class. This method is used to transform a regular NumPy array into a DistributedArray that is distributed and processed across multiple nodes or processes.

Moreover, the `DistributedArray` class provides also fundamental mathematical operations, such as element-wise addition, subtraction, multiplication, dot product, and an equivalent of the [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) function that operate in a distributed fashion, 
thus utilizing the efficiency of the MPI/NCC; protocols. This enables efficient computation and processing of large-scale distributed arrays.

## Running Tests
The MPI test scripts are located in the `tests` folder.
Use the following command to run the tests:
```
mpiexec -n <NUM_PROCESSES> pytest tests/ --with-mpi
```
where the `--with-mpi` option tells pytest to enable the `pytest-mpi` plugin, allowing the tests to utilize the MPI functionality.

Similarly, to run the NCCL test scripts in the `tests_nccl` folder, 
use the following command to run the tests:
```
mpiexec -n <NUM_PROCESSES> pytest tests_nccl/ --with-mpi
```

## Documentation 
The official documentation of Pylops-MPI is available [here](https://pylops.github.io/pylops-mpi/).
Visit the official docs to learn more about pylops-mpi.

## Contributors
* Rohan Babbar, rohanbabbar04
* Yuxi Hong, hongyx11
* Matteo Ravasi, mrava87
* Tharit Tangkijwanichakul, tharittk
