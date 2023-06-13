# pylops-mpi
[![pages-build-deployment](https://github.com/PyLops/pylops-mpi/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/PyLops/pylops-mpi/actions/workflows/pages/pages-build-deployment)
[![PyLops-MPI](https://github.com/PyLops/pylops-mpi/actions/workflows/build.yml/badge.svg)](https://github.com/PyLops/pylops-mpi/actions/workflows/build.yml)
![OS-support](https://img.shields.io/badge/OS-linux,osx-850A8B.svg)
[![Slack Status](https://img.shields.io/badge/chat-slack-green.svg)](https://pylops.slack.com)

## PyLops MPI
pylops-mpi is a Python library built on top of [PyLops](https://pylops.readthedocs.io/en/stable/), designed to enable distributed and parallel processing of 
large-scale linear algebra operations and computations.  

## Installation
To install pylops-mpi, you need to have MPI (Message Passing Interface) installed on your system.
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
 3. **Install pylops-mpi**: Once MPI is installed and verified, you can proceed to install `pylops-mpi`. 
   
   You can install with `make` and `pip`:
      ```
      make install
      ```
   
   You can install with `make` and `conda`:
      ```
      make install_conda
      ```
   
## Run Pylops-MPI
Once you have installed the prerequisites and pylops-mpi, you can run pylops-mpi using the `mpiexec` command. 
Here's an example on how to run the command:
```
mpiexec -n <NUM_PROCESSES> python <script_name>.py
```

## Example
The DistributedArray can be used to either broadcast or scatter the NumPy array across different 
ranks or processes.
```python
from pylops_mpi import DistributedArray, Partition

global_shape = (10, 5)

# Initialize a DistributedArray with partition set to Broadcast
dist_array_broadcast = DistributedArray(global_shape=global_shape,
                                        partition=Partition.BROADCAST)

# Initialize a DistributedArray with partition set to Scatter
dist_array_scatter = DistributedArray(global_shape=global_shape,
                                      partition=Partition.SCATTER)
```

Additionally, the DistributedArray can be used to scatter the array along any
specified axis.

```python
# Partition axis = 0
dist_array_0 = DistributedArray(global_shape=global_shape, 
                                partition=Partition.SCATTER, axis=0)

# Partition axis = 1
dist_array_1 = DistributedArray(global_shape=global_shape, 
                                partition=Partition.SCATTER, axis=1)
```

The DistributedArray class provides a `to_dist` class method that accepts a NumPy array as input and converts it into an 
instance of the `DistributedArray` class. This method is used to transform a regular NumPy array into a DistributedArray that can be distributed 
and processed across multiple nodes or processes.

```python
import numpy as np
np.random.seed(42)

dist_arr = DistributedArray.to_dist(x=np.random.normal(100, 100, global_shape), 
                                    partition=Partition.SCATTER, axis=0)
```
The DistributedArray also provides fundamental mathematical operations, like element-wise addition, subtraction, and multiplication, 
as well as dot product and the [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) function in a distributed fashion, 
thus utilizing the efficiency of the MPI protocol. This enables efficient computation and processing of large-scale distributed arrays.

## Running Tests
The test scripts are located in the tests folder.
Use the following command to run the tests:
```
mpiexec -n <NUM_PROCESSES> pytest --with-mpi
```
The `--with-mpi` option tells pytest to enable the `pytest-mpi` plugin, 
allowing the tests to utilize the MPI functionality.

## Documentation 
The official documentation of Pylops-MPI is available [here](https://pylops.github.io/pylops-mpi/).
Visit the official docs to learn more about pylops-mpi.

## Contributors
* Rohan Babbar, rohanbabbar04
* Matteo Ravasi, mrava87
