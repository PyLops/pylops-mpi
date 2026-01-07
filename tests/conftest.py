import sys
import pytest
from mpi4py import MPI

def pytest_itemcollected(item):
    """Append MPI rank to the test ID as it is collected."""
    rank = MPI.COMM_WORLD.Get_rank()
    item._nodeid += f"[Rank {rank}]"
