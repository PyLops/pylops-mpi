# pylops-mpi
[![pages-build-deployment](https://github.com/PyLops/pylops-mpi/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/PyLops/pylops-mpi/actions/workflows/pages/pages-build-deployment)
[![PyLops-MPI](https://github.com/PyLops/pylops-mpi/actions/workflows/build.yml/badge.svg)](https://github.com/PyLops/pylops-mpi/actions/workflows/build.yml)
![OS-support](https://img.shields.io/badge/OS-linux,win,osx-850A8B.svg)
[![Slack Status](https://img.shields.io/badge/chat-slack-green.svg)](https://pylops.slack.com)

## PyLops MPI
pylops-mpi is a Python library built on top of PyLops, designed to enable distributed and parallel processing of 
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
   
   You can install with `make` and `pip`
      ```
      make install
      ```
   
   You can install with `make` and `conda`
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



## Documentation 
The official documentation of Pylops-MPI is available [here](https://pylops.github.io/pylops-mpi/).
Visit the official docs to learn more about pylops-mpi.

## Contributors
* Matteo Ravasi, mrava87
* Rohan Babbar, rohanbabbar04